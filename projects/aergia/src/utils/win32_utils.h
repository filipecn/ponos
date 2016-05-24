#pragma once

#ifdef _WIN32

#define WIN32CONSOLE() \
// Open a new console window \
AllocConsole(); \
AttachConsole(GetCurrentProcessId()); \
//-- Associate std input/output with newly opened console window: \
freopen("CONIN$", "r", stdin); \
freopen("CONOUT$", "w", stdout); \
freopen("CONOUT$", "w", stderr);

#endif
