/* Fixed host_config.h to avoid recursive includes */
#ifndef FIXED_HOST_CONFIG_H
#define FIXED_HOST_CONFIG_H

/* Simplified version to avoid recursive includes */
#if defined(_MSC_VER)
#pragma message("Using fixed host_config.h")
#else
#warning "Using fixed host_config.h"
#endif

#endif /* FIXED_HOST_CONFIG_H */
