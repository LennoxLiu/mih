#ifndef MEMUSAGE_H__
#define MEMUSAGE_H__

#include <ios>
#include <iostream>
#include <fstream>
#include <string>

#if defined(_WIN32)
// Windows-specific implementation using Windows APIs.
#include <windows.h>
#include <psapi.h>
inline void process_mem_usage(double *vm_usage, double *resident_set)
{
   *vm_usage     = 0.0;
   *resident_set = 0.0;
   HANDLE process = GetCurrentProcess();
   PROCESS_MEMORY_COUNTERS pmc;
   if (GetProcessMemoryInfo(process, &pmc, sizeof(pmc))) {
      // Convert bytes to megabytes.
      *resident_set = (double)pmc.WorkingSetSize / (1024 * 1024);
      *vm_usage     = (double)pmc.PagefileUsage / (1024 * 1024);
   }
}
#else
// POSIX implementation (works on Linux)
#include <unistd.h>
inline void process_mem_usage(double *vm_usage, double *resident_set)
{
   using std::ios_base;
   using std::ifstream;
   using std::string;

   *vm_usage     = 0.0;
   *resident_set = 0.0;

   // Read from /proc/self/stat (Linux only)
   ifstream stat_stream("/proc/self/stat", ios_base::in);

   string pid, comm, state, ppid, pgrp, session, tty_nr;
   string tpgid, flags, minflt, cminflt, majflt, cmajflt;
   string utime, stime, cutime, cstime, priority, nice;
   string O, itrealvalue, starttime;

   unsigned long vsize;
   long rss;

   stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
               >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
               >> utime >> stime >> cutime >> cstime >> priority >> nice
               >> O >> itrealvalue >> starttime >> vsize >> rss;
   stat_stream.close();

   long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in KB
   *vm_usage     = vsize / 1024.0;
   *resident_set = rss * page_size_kb / 1024.0; // convert to MB if desired
}
#endif

#endif // MEMUSAGE_H__
