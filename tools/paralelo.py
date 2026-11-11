"""
Usar toda la PC: núcleos del procesador para los pasos que Open3D hace con uno solo.

simplificar(mesh, meta): reduce una malla grande a `meta` triángulos partiéndola en bloques
(por posición) que se reducen al mismo tiempo, uno por núcleo, y luego se vuelven a unir.
Los bordes entre bloques se protegen (casi no se mueven) para que la unión no deje huecos.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import open3d as o3d


def _prioridad_alta():
    """En Windows, los procesos de apoyo nacen con prioridad normal: se suben como el principal."""
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00008000)      # ABOVE_NORMAL_PRIORITY_CLASS
        except Exception:       # noqa: BLE001
            pass


def nucleos():
    """Núcleos lógicos disponibles (todos)."""
    return max(1, os.cpu_count() or 1)


def _reducir_bloque(args):
    V, F, C, meta = args
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    if C is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(C)
    if len(F) > meta:
        # boundary_weight alto: los bordes del bloque se quedan casi en su lugar
        m = m.simplify_quadric_decimation(int(meta), boundary_weight=1000.0)
    return (np.asarray(m.vertices).copy(), np.asarray(m.triangles).copy(),
            np.asarray(m.vertex_colors).copy() if m.has_vertex_colors() else None)


def _bloques(V, F, n):
    """Divide los triángulos en ~n bloques de tamaño parecido, cortando por el eje más largo cada vez."""
    cent = V[F].mean(1)
    grupos = [np.arange(len(F))]
    while len(grupos) < n:
        grupos.sort(key=len)
        g = grupos.pop()
        c = cent[g]
        eje = int(np.argmax(c.max(0) - c.min(0)))
        corte = np.median(c[:, eje])
        a, b = g[c[:, eje] <= corte], g[c[:, eje] > corte]
        if len(a) == 0 or len(b) == 0:
            grupos.append(g)
            break
        grupos += [a, b]
    return grupos


def simplificar(mesh, meta, log=None, min_paralelo=1_500_000):
    """Reduce `mesh` (TriangleMesh) a ~`meta` triángulos usando todos los núcleos. Devuelve una malla nueva."""
    n = len(mesh.triangles)
    if n <= meta:
        return mesh
    workers = nucleos()
    if workers < 2 or n < min_paralelo:
        return mesh.simplify_quadric_decimation(int(meta))
    t0 = time.time()
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    C = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    nb = min(64, workers * 2)               # más bloques que núcleos: se reparten mejor
    tareas = []
    for g in _bloques(V, F, nb):
        Fg = F[g]
        usados, inv = np.unique(Fg.reshape(-1), return_inverse=True)
        tareas.append((V[usados], inv.reshape(-1, 3).astype(np.int32),
                       None if C is None else C[usados], max(1, round(meta * len(g) / n))))
    partes = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_prioridad_alta) as ex:
        for r in ex.map(_reducir_bloque, tareas):
            partes.append(r)
    del tareas
    vs, fs, cs, off = [], [], [], 0
    for Vp, Fp, Cp in partes:
        vs.append(Vp)
        fs.append(Fp + off)
        if C is not None:
            cs.append(Cp if Cp is not None and len(Cp) == len(Vp) else np.full((len(Vp), 3), 0.6))
        off += len(Vp)
    out = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate(vs)),
                                    o3d.utility.Vector3iVector(np.concatenate(fs)))
    if C is not None:
        out.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(cs))
    out.merge_close_vertices(1e-6)          # soldar los bordes entre bloques
    out.remove_degenerate_triangles()
    out.remove_unreferenced_vertices()
    if log:
        log(f"  malla reducida en paralelo: {n:,} -> {len(out.triangles):,} triángulos con {workers} núcleos "
            f"({time.time() - t0:.0f} s)")
    return out


# ---------------------------------------------------------------- limpieza y suavizado rápidos
# Lo mismo que hacen las funciones de Open3D (un solo núcleo, varios segundos por millón de
# triángulos), escrito con operaciones vectoriales: decenas de veces más rápido.

def _arrays(mesh):
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    C = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    return V, F, C


def _malla(V, F, C):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
    if C is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(C)
    return m


def _compactar(V, F, C):
    usados, inv = np.unique(F.reshape(-1), return_inverse=True)
    return V[usados], inv.reshape(-1, 3), (None if C is None else C[usados])


def limpiar(mesh, min_triangles):
    """Une vértices repetidos, quita triángulos degenerados y pedazos sueltos de menos de `min_triangles`."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    V, F, C = _arrays(mesh)
    if len(F) == 0:
        return mesh
    # vértices exactamente repetidos (la malla del volumen los trae en los bordes de cada bloque)
    key = np.ascontiguousarray(V).view(np.dtype((np.void, V.dtype.itemsize * 3))).ravel()
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    V, C = V[first], (None if C is None else C[first])
    F = inv.reshape(-1)[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    # pedazos conectados
    n = len(V)
    a = np.r_[F[:, 0], F[:, 1]]
    b = np.r_[F[:, 1], F[:, 2]]
    g = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    tl = lab[F[:, 0]]
    cnt = np.bincount(tl)
    F = F[cnt[tl] >= min_triangles]
    if len(F) == 0:
        return _malla(V[:0], F, None if C is None else C[:0])
    out = _malla(*_compactar(V, F, C))
    out.compute_vertex_normals()
    return out


def taubin(mesh, iteraciones=10, lam=0.5, mu=-0.53):
    """Suavizado Taubin, igual que filter_smooth_taubin de Open3D: promedio de los vecinos pesado por
    1/distancia (recalculado en cada paso) y aplicado también a los colores."""
    from scipy.sparse import coo_matrix, csr_matrix
    V, F, C = _arrays(mesh)
    if len(F) == 0:
        return mesh
    n = len(V)
    a = np.r_[F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]]
    b = np.r_[F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]]
    A = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)).tocsr()   # vecinos únicos
    A.sum_duplicates()
    filas = np.repeat(np.arange(n), np.diff(A.indptr))
    cols = A.indices
    solo = np.diff(A.indptr) == 0
    X = V.copy()
    Y = None if C is None else C.copy()
    for _ in range(iteraciones):
        for f in (lam, mu):
            w = 1.0 / (np.linalg.norm(X[filas] - X[cols], axis=1) + 1e-12)
            W = csr_matrix((w, cols, A.indptr), shape=(n, n))
            tot = np.asarray(W.sum(1)).ravel()
            tot[solo] = 1
            D = (W @ X) / tot[:, None] - X
            D[solo] = 0
            if Y is not None:
                Dc = (W @ Y) / tot[:, None] - Y
                Dc[solo] = 0
                Y += f * Dc
            X += f * D
    out = _malla(X, F, None if Y is None else Y)
    out.compute_vertex_normals()
    return out


def _taubin_bloque(args):
    V, F, C, it, nucleo = args
    m = _malla(V, F, C)
    m = m.filter_smooth_taubin(number_of_iterations=it)
    Vs = np.asarray(m.vertices)[nucleo]
    Cs = np.asarray(m.vertex_colors)[nucleo] if C is not None else None
    return Vs, Cs


def taubin_paralelo(mesh, iteraciones=10, margen=0.15, log=None, min_paralelo=800_000):
    """Taubin de Open3D en bloques a la vez (uno por núcleo). Cada bloque lleva alrededor 15 cm extra de
    malla, más de lo que alcanza a "viajar" el suavizado, así que el resultado es igual al de hacerlo entero."""
    V, F, C = _arrays(mesh)
    workers = nucleos()
    if workers < 2 or len(F) < min_paralelo:
        m = mesh.filter_smooth_taubin(number_of_iterations=iteraciones)
        m.compute_vertex_normals()
        return m
    t0 = time.time()
    # bloques por posición de los vértices
    grupos = [np.arange(len(V))]
    while len(grupos) < min(64, workers * 2):
        grupos.sort(key=len)
        g = grupos.pop()
        p = V[g]
        eje = int(np.argmax(p.max(0) - p.min(0)))
        corte = np.median(p[:, eje])
        a, b = g[p[:, eje] <= corte], g[p[:, eje] > corte]
        if len(a) == 0 or len(b) == 0:
            grupos.append(g)
            break
        grupos += [a, b]
    tareas, destinos = [], []
    for g in grupos:
        lo, hi = V[g].min(0) - margen, V[g].max(0) + margen
        dentro = np.all((V >= lo) & (V <= hi), axis=1)
        tri = F[dentro[F].any(1)]
        usados, inv = np.unique(tri.reshape(-1), return_inverse=True)
        pos = np.full(len(V), -1, np.int64)
        pos[usados] = np.arange(len(usados))
        nucleo = pos[g]
        ok = nucleo >= 0                 # vértices sin triángulos: se quedan como están
        tareas.append((V[usados], inv.reshape(-1, 3), None if C is None else C[usados], iteraciones, nucleo[ok]))
        destinos.append(g[ok])
    X = V.copy()
    Y = None if C is None else C.copy()
    with ProcessPoolExecutor(max_workers=workers, initializer=_prioridad_alta) as ex:
        for dst, (Vs, Cs) in zip(destinos, ex.map(_taubin_bloque, tareas)):
            X[dst] = Vs
            if Y is not None:
                Y[dst] = Cs
    out = _malla(X, F, Y)
    out.compute_vertex_normals()
    if log:
        log(f"  suavizado en paralelo con {workers} núcleos ({time.time() - t0:.0f} s)")
    return out


def tapar_huecos(mesh, max_aristas=120, max_perimetro=0.6, log=None):
    """Tapa los huecos chicos de la malla (orillas cerradas de hasta `max_aristas` y `max_perimetro` m)
    con un abanico desde su centro. Los huecos grandes (zonas sin escanear) se dejan como están."""
    V, F, C = _arrays(mesh)
    if len(F) == 0:
        return mesh
    # aristas dirigidas de cada triángulo; una arista es orilla si su par opuesto no existe
    a = np.r_[F[:, 0], F[:, 1], F[:, 2]].astype(np.int64)
    b = np.r_[F[:, 1], F[:, 2], F[:, 0]].astype(np.int64)
    n = len(V)
    key = a * n + b
    rev = b * n + a
    orilla = ~np.isin(key, rev)
    ea, eb = a[orilla], b[orilla]
    # vértices con más de una arista de orilla saliente: nudo, no se tocan
    salidas = np.bincount(ea, minlength=n)
    sig = {}
    for x, y in zip(ea.tolist(), eb.tolist()):
        if salidas[x] == 1:
            sig[x] = y
    visto = set()
    nuevosV, nuevosC, nuevosF = [], [], []
    tapados = 0
    for inicio in list(sig.keys()):
        if inicio in visto:
            continue
        lazo = [inicio]
        x = inicio
        ok = False
        while len(lazo) <= max_aristas:
            y = sig.get(x)
            if y is None or (y in visto):
                break
            if y == inicio:
                ok = True
                break
            lazo.append(y)
            x = y
        visto.update(lazo)
        if not ok or len(lazo) < 3:
            continue
        P = V[lazo]
        per = np.linalg.norm(P - np.roll(P, -1, 0), axis=1).sum()
        if per > max_perimetro:
            continue
        idx = n + len(nuevosV)
        nuevosV.append(P.mean(0))
        if C is not None:
            nuevosC.append(C[lazo].mean(0))
        for i in range(len(lazo)):
            u, w = lazo[i], lazo[(i + 1) % len(lazo)]
            nuevosF.append((w, u, idx))          # orientación contraria a la orilla = misma cara que la malla
        tapados += 1
    if not tapados:
        return mesh
    V2 = np.vstack([V, np.array(nuevosV)])
    F2 = np.vstack([F, np.array(nuevosF, dtype=F.dtype)])
    C2 = None if C is None else np.vstack([C, np.array(nuevosC)])
    out = _malla(V2, F2, C2)
    out.compute_vertex_normals()
    if log:
        log(f"  {tapados:,} huecos chicos tapados")
    return out


def limpiar_orillas(mesh, anillos=2, min_triangles=1500, log=None):
    """Quita el fleco de las orillas (`anillos` filas de triángulos en cada borde abierto) y los pedazos
    sueltos chicos. En las orillas el LiDAR mezcla el objeto con lo de atrás: ahí salían los bordes
    blancos/"confeti" alrededor de los objetos."""
    V, F, C = _arrays(mesh)
    if len(F) == 0:
        return mesh
    n0 = len(F)
    for _ in range(anillos):
        a = np.r_[F[:, 0], F[:, 1], F[:, 2]].astype(np.int64)
        b = np.r_[F[:, 1], F[:, 2], F[:, 0]].astype(np.int64)
        n = len(V)
        orilla = ~np.isin(a * n + b, b * n + a)
        borde = np.zeros(n, bool)
        borde[a[orilla]] = True
        borde[b[orilla]] = True
        F = F[~borde[F].any(1)]
        if len(F) == 0:
            break
    out = limpiar(_malla(V, F, C), min_triangles)
    if log:
        log(f"  orillas limpias: {n0 - len(out.triangles):,} triángulos de fleco y pedazos sueltos fuera")
    return out


def superficie_continua(mesh, profundidad=None, max_dist=0.05, quitar_densidad=0.04, log=None):
    """Reconstrucción Poisson: una sola superficie continua y lisa que pasa por todos los puntos (como
    las apps de escaneo comerciales). Cierra huecos chicos y deja orillas limpias. Después se recorta
    todo lo que quedó lejos de lo medido (> max_dist) o con muy pocos puntos cerca (densidad baja),
    para no inventar superficies."""
    t0 = time.time()
    m = o3d.geometry.TriangleMesh(mesh)
    m.compute_vertex_normals()
    pcd = o3d.geometry.PointCloud(m.vertices)
    pcd.normals = m.vertex_normals
    if m.has_vertex_colors():
        pcd.colors = m.vertex_colors
    pcd = pcd.voxel_down_sample(0.004)
    if profundidad is None:
        # celdas de ~6 mm sin importar el tamaño del espacio (más grande = más profundidad, hasta 11)
        ext = float(np.max(pcd.get_max_bound() - pcd.get_min_bound()))
        profundidad = int(np.clip(np.ceil(np.log2(max(ext, 0.1) / 0.006)), 8, 11))
    p, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=profundidad, n_threads=nucleos())
    dens = np.asarray(dens)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    d = sc.compute_distance(o3d.core.Tensor(np.asarray(p.vertices).astype(np.float32))).numpy()
    malo = (d > max_dist) | (dens < np.quantile(dens, quitar_densidad))
    p.remove_vertices_by_mask(malo)
    out = limpiar(p, 1500)
    if log:
        log(f"  superficie continua: {len(out.triangles):,} triángulos ({time.time() - t0:.0f} s)")
    return out
