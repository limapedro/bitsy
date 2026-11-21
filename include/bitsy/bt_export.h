#ifndef BITSY_BT_EXPORT_H
#define BITSY_BT_EXPORT_H

/* Windows needs an explicit export/import decision for the shared library. */
#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(BT_BUILD_SHARED)
#    define BT_API __declspec(dllexport)
#  else
#    define BT_API __declspec(dllimport)
#  endif
#else
#  define BT_API __attribute__((visibility("default")))
#endif

#endif /* BITSY_BT_EXPORT_H */
