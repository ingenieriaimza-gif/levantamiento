; Instalador de Levantamiento 3D para Windows (Inno Setup 6)
; Se compila en GitHub Actions: iscc /DMyVersion=<n> installer\levantamiento.iss

#define MyAppName "Levantamiento 3D"
#ifndef MyVersion
  #define MyVersion "0"
#endif

[Setup]
AppId={{8F3A2B6C-5D1E-4C7A-9B2F-3E4D5C6B7A81}
AppName={#MyAppName}
AppVersion={#MyVersion}
AppVerName={#MyAppName} (versión {#MyVersion})
AppPublisher=Levantamiento 3D
; en la carpeta del usuario para que el programa pueda actualizarse solo
DefaultDirName={localappdata}\Programs\Levantamiento3D
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=admin
UsedUserAreasWarning=no
OutputDir=..\dist
OutputBaseFilename=Levantamiento3D_Setup
SetupIconFile=icono.ico
UninstallDisplayIcon={app}\pc\web\icono.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "Crear acceso directo en el escritorio"; GroupDescription: "Accesos directos:"

[Files]
Source: "..\dist\app\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "icono.ico"; DestDir: "{app}\pc\web"; Flags: ignoreversion
Source: "..\dist\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "-I ""{app}\pc\lanzador.py"""; WorkingDir: "{app}"; IconFilename: "{app}\pc\web\icono.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "-I ""{app}\pc\lanzador.py"""; WorkingDir: "{app}"; IconFilename: "{app}\pc\web\icono.ico"; Tasks: desktopicon

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Instalando componentes de Microsoft Visual C++…"; Flags: waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""Levantamiento 3D"""; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""Levantamiento 3D"" dir=in action=allow protocol=TCP localport=8765 profile=any"; StatusMsg: "Permitiendo la conexión del iPhone en el firewall…"; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""Levantamiento 3D"" dir=in action=allow protocol=UDP localport=5353 profile=any"; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""Levantamiento 3D"" dir=in action=allow program=""{app}\python\pythonw.exe"" profile=any"; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""Levantamiento 3D"" dir=in action=allow program=""{app}\python\python.exe"" profile=any"; Flags: runhidden waituntilterminated
Filename: "{app}\python\pythonw.exe"; Parameters: "-I ""{app}\pc\lanzador.py"""; WorkingDir: "{app}"; Description: "Abrir Levantamiento 3D"; Flags: postinstall nowait skipifsilent

[UninstallRun]
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""Levantamiento 3D"""; Flags: runhidden; RunOnceId: "QuitarFirewall"

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Messages]
spanish.WelcomeLabel2=Esto instalará [name] en tu computadora: el programa que recibe los escaneos de la app 3D del iPhone por red local, los muestra en vivo y los procesa para Navisworks y SolidWorks.%n%nTus escaneos se guardan en la carpeta Levantamientos3D de tu usuario y no se borran al desinstalar.
