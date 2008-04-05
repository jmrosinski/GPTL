dnl Process this file with autoconf to produce a configure script.
AC_INIT(private.h)
AC_LANG(C)
echo "This script provides suggestions for settings to apply in macros.make"
echo "You can pass things like FC=gfortran or CC=pathcc to it to override defaults."
AC_PROG_MAKE_SET
AC_PROG_CC()
echo "Assuming C compiler is $CC"
AC_PROG_FC()
echo "Assuming Fortran compiler is $FC"
dnl This barfs when CC is pgcc and F77 pgf90
AC_FC_FUNC(z_zz,z_zz)
case $z_zz in
    z_zz__)     echo "Fortran name mangling: -DFORTRANDOUBLEUNDERSCORE" ;;
    z_zz_)      echo "Fortran name mangling: -DFORTRANUNDERSCORE" ;;
    Z_ZZ)       echo "Fortran name mangling: -DFORTRANCAPS" ;;
    *)          echo "Fortran name mangling: none" ;;
esac

dnl For proper handling of const and inline
AC_C_INLINE
echo "Inlining: -Dinline=$ac_cv_c_inline"
dnl AC_FUNC_VPRINTF
