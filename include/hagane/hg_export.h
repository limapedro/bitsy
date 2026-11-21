#ifndef HAGANE_HG_EXPORT_H
#define HAGANE_HG_EXPORT_H

/* Symbol visibility for the Hagane C ABI. */
#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HG_BUILD_SHARED)
#    define HG_API __declspec(dllexport)
#  else
#    define HG_API __declspec(dllimport)
#  endif
#else
#  define HG_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#  define HG_EXTERN_C extern "C"
#else
#  define HG_EXTERN_C
#endif

#endif /* HAGANE_HG_EXPORT_H */
