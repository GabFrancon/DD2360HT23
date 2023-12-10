#include "Timing.h"

double cpuSecond()
{
    return (double)clock() / CLOCKS_PER_SEC;
}
