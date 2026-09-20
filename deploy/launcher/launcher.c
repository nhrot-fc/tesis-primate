/* Lanzador del paquete de Windows. Deja el entorno autocontenido (nada sale a internet ni se
 * escribe fuera de la carpeta) y arranca el Python embebido con el script que toca, pasando
 * los argumentos tal cual: una carpeta o un audio arrastrados sobre el .exe llegan al visor.
 *
 * Lo compila deploy/build_windows.py dos veces, con zig desde Linux:
 *   viewer.exe   -DGUI   pythonw.exe src\main.py     sin consola; si falla avisa con un cuadro
 *   detect.exe             python.exe  src\detect.py   con consola; devuelve el código del script
 */
#include <wchar.h>
#include <windows.h>

#ifdef GUI
#define INTERPRETER L"python\\pythonw.exe"
#define SCRIPT L"src\\main.py"
#else
#define INTERPRETER L"python\\python.exe"
#define SCRIPT L"src\\detect.py"
#endif

#define TITLE L"Primate Vocalization Detector"
#define ROOT_MAX MAX_PATH
#define COMMAND_MAX 32768

static void complain(const wchar_t *message) {
#ifdef GUI
    MessageBoxW(NULL, message, TITLE, MB_OK | MB_ICONERROR);
#else
    DWORD written;
    WriteConsoleW(GetStdHandle(STD_ERROR_HANDLE), message, (DWORD)wcslen(message), &written, NULL);
    WriteConsoleW(GetStdHandle(STD_ERROR_HANDLE), L"\r\n", 2, &written, NULL);
#endif
}

/* Los argumentos con los que se llamó al .exe, sin el propio .exe (entre comillas o no). */
static const wchar_t *arguments(void) {
    const wchar_t *line = GetCommandLineW();
    if (*line == L'"') {
        line++;
        while (*line && *line != L'"') line++;
        if (*line == L'"') line++;
    } else {
        while (*line && *line != L' ' && *line != L'\t') line++;
    }
    while (*line == L' ' || *line == L'\t') line++;
    return line;
}

static void set_path_variable(const wchar_t *name, const wchar_t *root, const wchar_t *relative) {
    wchar_t value[ROOT_MAX * 2];
    wcscpy(value, root);
    wcscat(value, relative);
    SetEnvironmentVariableW(name, value);
}

static int launch(void) {
    /* Carpeta del .exe, con la barra final */
    wchar_t root[ROOT_MAX];
    DWORD length = GetModuleFileNameW(NULL, root, ROOT_MAX);
    if (length == 0 || length >= ROOT_MAX) {
        complain(L"Could not find where this program lives.");
        return 1;
    }
    for (wchar_t *p = root + length; p > root; p--) {
        if (p[-1] == L'\\') { *p = 0; break; }
    }

    wchar_t interpreter[ROOT_MAX * 2];
    wcscpy(interpreter, root);
    wcscat(interpreter, INTERPRETER);
    if (GetFileAttributesW(interpreter) == INVALID_FILE_ATTRIBUTES) {
        complain(L"python\\ is missing next to this program.\n"
                 L"Unzip the whole package into one folder and run it from there.");
        return 1;
    }

    /* Igual que lo que traía python\entorno.bat: sin red, caches dentro del paquete */
    SetEnvironmentVariableW(L"HF_HUB_OFFLINE", L"1");
    SetEnvironmentVariableW(L"TRANSFORMERS_OFFLINE", L"1");
    SetEnvironmentVariableW(L"HF_HUB_DISABLE_TELEMETRY", L"1");
    SetEnvironmentVariableW(L"YOLO_OFFLINE", L"1");
    SetEnvironmentVariableW(L"PYTHONUTF8", L"1");
    set_path_variable(L"HF_HOME", root, L"cache\\hf");
    set_path_variable(L"YOLO_CONFIG_DIR", root, L"cache\\ultralytics");
    set_path_variable(L"MPLCONFIGDIR", root, L"cache\\matplotlib");

    /* "python\pythonw.exe" "src\main.py" <argumentos originales> */
    static wchar_t command[COMMAND_MAX];
    const wchar_t *rest = arguments();
    if (wcslen(interpreter) + wcslen(root) + wcslen(SCRIPT) + wcslen(rest) + 8 >= COMMAND_MAX) {
        complain(L"Too many arguments.");
        return 1;
    }
    wcscpy(command, L"\"");
    wcscat(command, interpreter);
    wcscat(command, L"\" \"");
    wcscat(command, root);
    wcscat(command, SCRIPT);
    wcscat(command, L"\"");
    if (*rest) {
        wcscat(command, L" ");
        wcscat(command, rest);
    }

    STARTUPINFOW startup;
    PROCESS_INFORMATION process;
    ZeroMemory(&startup, sizeof(startup));
    startup.cb = sizeof(startup);
    ZeroMemory(&process, sizeof(process));
    if (!CreateProcessW(NULL, command, NULL, NULL, TRUE, 0, NULL, root, &startup, &process)) {
        complain(L"Could not start the embedded Python.");
        return 1;
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hProcess);
    CloseHandle(process.hThread);
#ifdef GUI
    if (code != 0) {
        complain(L"The viewer closed with an error.\n"
                 L"The reason is in viewer.log, next to this program.");
    }
#endif
    return (int)code;
}

#ifdef GUI
int WINAPI WinMain(HINSTANCE instance, HINSTANCE previous, LPSTR line, int show) {
    (void)instance; (void)previous; (void)line; (void)show;
    return launch();
}
#else
int main(void) {
    return launch();
}
#endif
