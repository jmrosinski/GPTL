dnl Process this file with autoconf to produce a configure script.
AC_INIT(private.h)
AC_LANG(C)

AC_PROG_MAKE_SET
AC_PROG_CC()
#AC_PROG_FC()
dnl This barfs when CC is pgcc and F77 pgf90
AC_FC_FUNC(z_zz,z_zz)
case $z_zz in
    z_zz__)     echo FORTRANDOUBLEUNDERSCORE ;;
    z_zz_)      echo FORTRANUNDERSCORE ;;
    Z_ZZ)       echo FORTRANCAPS ;;
esac

dnl For proper handling of const and inline
AC_C_INLINE
dnl Checks for library functions.
AC_FUNC_VPRINTF
