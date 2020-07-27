
#pragma once


#define ENABLE_ASSERTIONS 1

#if ENABLE_ASSERTIONS
    #undef assert
    #define assert(COND) if (!(COND)) { \
        printf("'%s' failed at %s:%d\n", #COND, __FILE__, __LINE__); \
        *(int*) 0 = 0; \
    }

#endif
