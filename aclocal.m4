dnl $Id: aclocal.m4,v 1.1 2000-12-28 00:38:05 rosinski Exp $
dnl UD macros for netcdf configure


dnl Convert a string to all uppercase.
dnl
define([uppercase],
[translit($1, abcdefghijklmnopqrstuvwxyz, ABCDEFGHIJKLMNOPQRSTUVWXYZ)])

dnl
dnl Check for a Standard C compiler.  Prefer a native one over the
dnl GNU one to reduce the chance that the environment variable LIBS
dnl will have to be set to reference the GNU C runtime library.
dnl
AC_DEFUN(UD_PROG_CC,
[
    # Because we must have a C compiler, we treat an unset CC
    # the same as an empty CC.
    case "${CC}" in
	'')
	    case `uname` in
		ULTRIX)
		    # The native ULTRIX C compiler isn't standard.
		    ccs='gcc cc'
		    ;;
		Linux)
		    # pgcc is threaded so we prefer it
		    ccs='pgcc cc'
		    ;;
		*)
		    # xlc is before c89 because AIX's sizeof(long long)
		    # differs between the two.
		    #
		    ccs='xlc c89 acc cc gcc'
		    ;;
	    esac
	    for cc in $ccs; do
		AC_CHECK_PROG(CC, $cc, $cc)
		case "$CC" in
		    '') ;;
		    *)  break
			;;
		esac
	    done
	    case "${CC}" in
		'')
		    AC_MSG_ERROR("Could not find C compiler")
		    ;;
	    esac
	    ;;
	*)
	    AC_CHECKING(user-defined C compiler \"$CC\")
	    ;;
    esac
    #
    # On some systems, a discovered compiler nevertheless won't
    # work (due to licensing, for example); thus, we check the
    # compiler with a test program.
    # 
    AC_MSG_CHECKING(C compiler)
    AC_TRY_COMPILE(, ,
	AC_MSG_RESULT(works),
	AC_MSG_ERROR($CC failed to compile test program))
    AC_SUBST(CC)
    case "$CC" in
	*gcc*)
	    GCC=yes		# Expected by autoconf(1) macros
	    ;;
    esac
    case `uname -sr` in
	'HP-UX A.09'*)
	    AC_DEFINE(_HPUX_SOURCE)
	    ;;
    esac
])

dnl Check for Fortran compiler.
dnl
AC_DEFUN(UD_PROG_FC,
[
    AC_BEFORE([UD_FORTRAN_TYPES])
    case `uname -sr` in
	AIX*)
	    forts="xlf90 xlf"
	    ;;
	BSD/OS*)
	    forts="f90 f77 fort77 g77"
	    ;;
	HP-UX*)
	    # f77(1) doesn't have the -L option.
	    forts=fort77
	    flibs=-lU77
	    ;;
	IRIX*)
	    forts="f90 f77"
	    ;;
	Linux*)
	    forts="pgf90 f77 g77 fort77"
	    ;;
	OSF1*)
	    # The use of f90(1) results in the following for
	    # an unknown reason (`make' works in the fortran/
	    # directory):
	    # f90 -c -I../libsrc ftest.F 
	    # Last chance handler: pc = 0xa971b8, sp = 0x3fece0, ra = 0xa971b8
	    # Last chance handler: internal exception: unwinding
	    forts="f90 f77"
	    ;;
	'SunOS 4'*)
	    forts='f90 g77 fort77'
	    ;;
	'SunOS 5'*)
	    # SunOS's f90(1) has problems passing a C `char'
	    # as a Fortran `integer*1' => use f77(1)
	    forts="f90"
	    ;;
	sn*|UNICOS*)
	    forts="f90 fort77 cf77 f77 g77"
	    ;;
	*)
	    forts="f90 xlf90 fort77 ghf77 f77 cf77 g77 xlf"
	    ;;
    esac

    FFLAGS="${FFLAGS}"
    FLIBS="${FLIBS} ${flibs}"

    case "${FC+set}" in
	set)
	    case "$FC" in
		'')
		    AC_MSG_WARN(no Fortran compiler)
		    ;;
		*)
		    AC_MSG_CHECKING(user-defined Fortran compiler \"$FC\")
		    cat <<EOF >conftest.f
                        CALL FOO
                        END
EOF
		    doit='$FC -c ${FFLAGS} conftest.f'
		    if AC_TRY_EVAL(doit); then
			AC_MSG_RESULT(works)
		    else
			AC_MSG_WARN($FC failed to compile test program)
			FC=
		    fi
		    rm -f conftest.*
		    ;;
	    esac
	    ;;
	*)
	    for fc in $forts; do
		AC_CHECK_PROG(FC, $fc, $fc)
		case "${FC}" in
		    '')
			;;
		    *)
			#
                        # On some systems, a discovered compiler
                        # nevertheless won't work (due to licensing,
                        # for example); thus, we check the compiler
                        # with a test program.
			# 
			cat <<EOF >conftest.f
                            CALL FOO
                            END
EOF
			doit='$FC -c ${FFLAGS} conftest.f'
			if AC_TRY_EVAL(doit); then
			    break
			else
			    AC_MSG_WARN($FC failed to compile test program)
			    unset FC
			    unset ac_cv_prog_FC
			fi
			;;
		esac
	    done
	    rm -f conftest.*
	    case "${FC}" in
		'') AC_MSG_WARN("Could not find working Fortran compiler")
		    AC_MSG_WARN(Setting FC to the empty string)
		    ;;
	    esac
	    ;;
    esac
    case "${FC}" in
	'')
	    AC_MSG_WARN("The Fortran interface will not be built")
	    ;;
    esac
    AC_SUBST(FC)
    AC_SUBST(FFLAGS)
    AC_SUBST(FLIBS)
    #
    # Set the make(1) macro for compiling a .F file.
    #
    AC_MSG_CHECKING(for Fortran .F compiler)
    AC_MSG_RESULT($COMPILE_F)
    case "${COMPILE_F-unset}" in
    unset)
	case "${FC}" in
	'')
	    COMPILE_F=
	    ;;
	*)
	    AC_MSG_CHECKING(if Fortran compiler handles *.F files)
	    cat >conftest.h <<\EOF
#define J 1
EOF
	    cat >conftest.F <<\EOF
#include "conftest.h"
#define N 5
              real r(J,N)
              end
EOF
	    doit='$FC -o conftest ${FFLAGS} conftest.F ${FLIBS}'
	    if AC_TRY_EVAL(doit); then
		COMPILE_F='$(COMPILE.f) $(FPPFLAGS)'
		AC_MSG_RESULT(yes)
	    else
		COMPILE_F=
		AC_MSG_RESULT(no)
	    fi
	    rm -f conftest*
	    ;;
	esac
	;;
    esac
    case "${COMPILE_F-}" in
    '') UD_PROG_FPP;;
    esac
    AC_SUBST(COMPILE_F)
    FPPFLAGS=${FPPFLAGS-}
    AC_SUBST(FPPFLAGS)
])


dnl Check for Fortran preprocessor.
dnl
AC_DEFUN(UD_PROG_FPP,
[
    AC_MSG_CHECKING(for Fortran preprocessor)
    case "$FPP" in
    '')
	AC_REQUIRE([AC_PROG_CPP])
	FPP="$CPP"
	;;
    esac
    AC_MSG_RESULT($FPP)
    AC_SUBST(FPP)
])


dnl Check for a Fortran type equivalent to a C type.
dnl
dnl UD_CHECK_FORTRAN_CTYPE(v3forttype, v2forttype, ctype, min, max)
dnl
AC_DEFUN(UD_CHECK_FORTRAN_CTYPE,
[
    AC_MSG_CHECKING(for Fortran-equivalent to C \"$3\")
    cat >conftest.f <<EOF
        subroutine sub(values, minval, maxval)
        implicit        none
        $2              values(5), minval, maxval
        minval = values(2)
        maxval = values(4)
        if (values(2) .ge. values(4)) then
            minval = values(4)
            maxval = values(2)
        endif
        end
EOF
    doit='$FC -c ${FFLAGS} conftest.f'
    if AC_TRY_EVAL(doit); then
	mv conftest.o conftestf.o
	cat >conftest.c <<EOF
#include <limits.h>
#include <float.h>
void main()
{
$3		values[[]] = {0, $4, 0, $5, 0};
$3		minval, maxval;
void	$FCALLSCSUB($3*, $3*, $3*);
$FCALLSCSUB(values, &minval, &maxval);
exit(!(minval == $4 && maxval == $5));
}
EOF
	doit='$CC -o conftest ${CPPFLAGS} ${CFLAGS} ${LDFLAGS} conftest.c conftestf.o ${LIBS}'
	if AC_TRY_EVAL(doit); then
	    doit=./conftest
	    if AC_TRY_EVAL(doit); then
		AC_MSG_RESULT($2)
		$1=$2
		AC_DEFINE_UNQUOTED($1,$2)
	    else
		AC_MSG_RESULT(no equivalent type)
		unset $1
	    fi
	else
	    AC_MSG_ERROR(Could not compile-and-link conftest.c and conftestf.o)
	fi
    else
	AC_MSG_ERROR(Could not compile conftest.f)
    fi
    rm -f conftest*
])


dnl Check for the name format of a Fortran-callable C routine.
dnl
dnl UD_CHECK_FCALLSCSUB
AC_DEFUN([UD_CHECK_FCALLSCSUB],
[
    AC_REQUIRE([UD_PROG_FC])
    case "$FC" in
	'') ;;
	*)  AC_BEFORE([UD_CHECK_FORTRAN_CTYPE])
	    AC_BEFORE([UD_CHECK_CTYPE_FORTRAN])
	    AC_MSG_CHECKING(for C-equivalent to Fortran routine \"SUB\")
	    cat >conftest.f <<\EOF
              call sub()
              end
EOF
	    doit='$FC -c ${FFLAGS} conftest.f'
	    if AC_TRY_EVAL(doit); then
		FCALLSCSUB=`nm conftest.o | awk '
		    /SUB_/{print "SUB_";exit}
		    /SUB/ {print "SUB"; exit}
		    /sub_/{print "sub_";exit}
		    /sub/ {print "sub"; exit}'`
		case "$FCALLSCSUB" in
		    '') AC_MSG_ERROR(not found)
			;;
		    *)  AC_MSG_RESULT($FCALLSCSUB)
			;;
		esac
	    else
		AC_MSG_ERROR(Could not compile conftest.f)
	    fi
	    rm -f conftest*
	    ;;
    esac
])


dnl Check for a C type equivalent to a Fortran type.
dnl
dnl UD_CHECK_CTYPE_FORTRAN(ftype, ctypes, fmacro_root)
dnl
AC_DEFUN(UD_CHECK_CTYPE_FORTRAN,
[
    cat >conftestf.f <<EOF
           $1 values(4)
           data values /-1, -2, -3, -4/
           call sub(values)
           end
EOF
    for ctype in $2; do
	AC_MSG_CHECKING(if Fortran \"$1\" is C \"$ctype\")
	cat >conftest.c <<EOF
	    void $FCALLSCSUB(values)
		$ctype values[[4]];
	    {
		exit(values[[1]] != -2 || values[[2]] != -3);
	    }
EOF
	doit='$CC -c ${CPPFLAGS} ${CFLAGS} conftest.c'
	if AC_TRY_EVAL(doit); then
	    doit='$FC ${FFLAGS} -c conftestf.f'
	    if AC_TRY_EVAL(doit); then
	        doit='$FC -o conftest ${FFLAGS} ${FLDFLAGS} conftestf.o conftest.o ${LIBS}'
	        if AC_TRY_EVAL(doit); then
		    doit=./conftest
		    if AC_TRY_EVAL(doit); then
		        AC_MSG_RESULT(yes)
		        cname=`echo $ctype | tr ' abcdefghijklmnopqrstuvwxyz' \
			    _ABCDEFGHIJKLMNOPQRSTUVWXYZ`
		        AC_DEFINE_UNQUOTED(NF_$3[]_IS_C_$cname)
		        break
		    else
		        AC_MSG_RESULT(no)
		    fi
	        else
		    AC_MSG_ERROR(Could not link conftestf.o and conftest.o)
	        fi
	    else
		AC_MSG_ERROR(Could not compile conftestf.f)
	    fi
	else
	    AC_MSG_ERROR(Could not compile conftest.c)
	fi
    done
    rm -f conftest*
])


dnl Get information about Fortran data types.
dnl
AC_DEFUN([UD_FORTRAN_TYPES],
[
    AC_REQUIRE([UD_PROG_FC])
    case "$FC" in
    '')
	;;
    *)
	AC_REQUIRE([UD_CHECK_FCALLSCSUB])
	UD_CHECK_FORTRAN_TYPE(NF_INT1_T, byte integer*1 "integer(kind(1))")
	UD_CHECK_FORTRAN_TYPE(NF_INT2_T, integer*2 "integer(kind(2))")

	case "${NF_INT1_T}" in
	    '') ;;
	    *)  UD_CHECK_CTYPE_FORTRAN($NF_INT1_T, "signed char", INT1)
		UD_CHECK_CTYPE_FORTRAN($NF_INT1_T, "short", INT1)
		UD_CHECK_CTYPE_FORTRAN($NF_INT1_T, "int", INT1)
		UD_CHECK_CTYPE_FORTRAN($NF_INT1_T, "long", INT1)
		;;
	esac
	case "${NF_INT2_T}" in
	    '') ;;
	    *)  UD_CHECK_CTYPE_FORTRAN($NF_INT2_T, short, INT2)
		UD_CHECK_CTYPE_FORTRAN($NF_INT2_T, int, INT2)
		UD_CHECK_CTYPE_FORTRAN($NF_INT2_T, long, INT2)
		;;
	esac
	UD_CHECK_CTYPE_FORTRAN(integer, int long, INT)
	UD_CHECK_CTYPE_FORTRAN(real, float double, REAL)
	UD_CHECK_CTYPE_FORTRAN(doubleprecision, double float, DOUBLEPRECISION)

	UD_CHECK_FORTRAN_NCTYPE(NCBYTE_T, byte integer*1 integer, byte)

	UD_CHECK_FORTRAN_NCTYPE(NCSHORT_T, integer*2 integer, short)
dnl	UD_CHECK_FORTRAN_CTYPE(NF_SHORT_T, $NCSHORT_T, short, SHRT_MIN, SHRT_MAX)

dnl	UD_CHECK_FORTRAN_NCTYPE(NCLONG_T, integer*4 integer, long)
dnl	UD_CHECK_FORTRAN_CTYPE(NF_INT_T, integer, int, INT_MIN, INT_MAX)

dnl	UD_CHECK_FORTRAN_NCTYPE(NCFLOAT_T, real*4 real, float)
dnl	UD_CHECK_FORTRAN_CTYPE(NF_FLOAT_T, $NCFLOAT_T, float, FLT_MIN, FLT_MAX)

dnl	UD_CHECK_FORTRAN_NCTYPE(NCDOUBLE_T, real*8 doubleprecision real, double)
dnl	UD_CHECK_FORTRAN_CTYPE(NF_DOUBLE_T, $NCDOUBLE_T, double, DBL_MIN, DBL_MAX)
	;;
    esac
])


dnl Check for the math library.
dnl
AC_DEFUN(UD_CHECK_LIB_MATH,
[
    AC_CHECKING(for math library)
    case "${MATHLIB}" in
	'')
	    AC_CHECK_LIB(c, tanh, MATHLIB=,
		    AC_CHECK_LIB(m, tanh, MATHLIB=-lm, MATHLIB=))
	    ;;
	*)
	    AC_MSG_RESULT($MATHLIB (user defined))
	    ;;
    esac
    AC_SUBST(MATHLIB)
])
dnl----------------------------End of netcdf--------------------------------
dnl JR: Taken from mpich
dnl
dnl Fortran runtime for Fortran/C linking
dnl On suns, try
dnl FC_LIB          =/usr/local/lang/SC2.0.1/libM77.a \ 
dnl              /usr/local/lang/SC2.0.1/libF77.a -lm \
dnl              /usr/local/lang/SC2.0.1/libm.a \
dnl              /usr/local/lang/SC2.0.1/libansi.a
dnl
dnl AIX requires -bI:/usr/lpp/xlf/lib/lowsys.exp
dnl ------------------------------------------------------------------------
dnl
dnl Get the format of Fortran names.  Uses F77, FFLAGS, and sets WDEF.
dnl If the test fails, sets NOF77 to 1, HAS_FORTRAN to 0
dnl
AC_DEFUN(PAC_GET_FORTNAMES,[
  AC_REQUIRE([UD_PROG_FC])
   # Check for strange behavior of Fortran.  For example, some FreeBSD
   # systems use f2c to implement f77, and the version of f2c that they 
   # use generates TWO (!!!) trailing underscores
   # Currently, WDEF is not used but could be...
   #
   # Eventually, we want to be able to override the choices here and
   # force a particular form.  This is particularly useful in systems
   # where a Fortran compiler option is used to force a particular
   # external name format (rs6000 xlf, for example).
   cat > confftest.F <<EOF
       subroutine t_startf(xxx)
       integer xxx
       xxx = 1
       return
       end
EOF
   $FC $FFLAGS -c confftest.F > /dev/null 2>&1
   if test ! -s confftest.o ; then
        echo "Unable to test Fortran compiler"
        echo "(compiling a test program failed to produce an "
        echo "object file)."
	NOFC=1
        HAS_FORTRAN=0
   elif test -z "$FORTRANNAMES" ; then
    # We have to be careful here, since the name may occur in several
    # forms.  We try to handle this by testing for several forms
    # directly.
    if test $arch_CRAY ; then
     # Cray doesn't accept -a ...
     nameform1=`strings confftest.o | grep t_startf_  | sed -n -e '1p'`
     nameform2=`strings confftest.o | grep T_STARTF   | sed -n -e '1p'`
     nameform3=`strings confftest.o | grep t_startf   | sed -n -e '1p'`
     nameform4=`strings confftest.o | grep t_startf__ | sed -n -e '1p'`
    else
     nameform1=`strings -a confftest.o | grep t_startf_  | sed -n -e '1p'`
     nameform2=`strings -a confftest.o | grep T_STARTF   | sed -n -e '1p'`
     nameform3=`strings -a confftest.o | grep t_startf   | sed -n -e '1p'`
     nameform4=`strings -a confftest.o | grep t_startf__ | sed -n -e '1p'`
    fi
    /bin/rm -f confftest.F confftest.o
    if test -n "$nameform4" ; then
	AC_MSG_RESULT(Fortran externals are lower case and have 1 or 2 trailing underscores)
	FORTRANNAMES="FORTRANDOUBLEUNDERSCORE"
    elif test -n "$nameform1" ; then
        # We don't set this in CFLAGS; it is a default case
        AC_MSG_RESULT([Fortran externals have a trailing underscore and are lowercase])
	FORTRANNAMES="FORTRANUNDERSCORE"
    elif test -n "$nameform2" ; then
	AC_MSG_RESULT(Fortran externals are upper case)
	FORTRANNAMES="FORTRANCAPS" 
    elif test -n "$nameform3" ; then
	AC_MSG_RESULT(Fortran externals are lower case)
	FORTRANNAMES="FORTRANNOUNDERSCORE"
    else
	echo "Unable to determine the form of Fortran external names"
	echo "Make sure that the compiler $FC can be run on this system"
#	echo "If you have problems linking, try using the -nof77 option"
#        echo "to configure and rebuild MPICH."
	echo "Turning off Fortran (-nof77 being assumed)."
	NOFC=1
        HAS_FORTRAN=0
    fi
#   case $nameform in 
#       SUB | _SUB)
#	echo "Fortran externals are uppercase"     
#	WDEF=-DFORTRANCAPS 
#	;;
#       sub_ | _sub_)   
#	 # We don't set this in CFLAGS; it is a default case
#        echo "Fortran externals have a trailing underscore and are lowercase"
#	WDEF=-DFORTRANUNDERSCORE ;;
#
#       sub | _sub)     
#	echo "Fortran externals are lower case"
#	WDEF=-DFORTRANNOUNDERSCORE 
#	;;
#
#           # Fortran no underscore is the "default" case for the wrappers; 
#	   # having this defined allows us to have an explicit test, 
#	   # particularly since the most common UNIX case is FORTRANUNDERSCORE
#       sub__ | _sub__)  
#	echo "Fortran externals are lower case and have 1 or 2 trailing underscores"
#	WDEF=-DFORTRANDOUBLEUNDERSCORE
#        ;;
#
#       *)
#	echo "Unable to determine the form of Fortran external names"
#	echo "If you have problems linking, try using the -nof77 option"
#        echo "to configure and rebuild MPICH."
#	NOFC=1
#        HAS_FORTRAN=0
#	;;
#   esac
    fi
    if test -n "$FORTRANNAMES" ; then
        WDEF="-D$FORTRANNAMES"
    fi
    AC_SUBST(WDEF)
    ])dnl

AC_DEFUN(UD_GET_ARCH,
[
# JR: Taken from mpich configure.in
# Check that an ARCH was set
# If it wasn't set, try to guess using "bin/tarch"

if test -z "$ARCH" ; then 
    # First check for some special cases
    if test -n "$device_t3d" ; then 
	ARCH=cray_t3d
        arch_cray_t3d=1
    fi
fi
if test -z "$ARCH" -a -x ./tarch ; then
    AC_MSG_CHECKING(for architecture)
    ARCH=`./tarch | sed s/-/_/g`
    if test -z "$ARCH" ; then
       AC_MSG_RESULT(Unknown!)
       echo "Error: Couldn't guess target architecture, you must"
       echo "       set an architecture type with -arch=<value>"
       exit 1
    fi
    eval "arch_$ARCH=1"
    AC_MSG_RESULT($ARCH)
fi
if test -n "$arch_sgi" ; then
    arch_IRIX=1
    ARCH=IRIX
fi
if test -n "$arch_IRIX64" ; then
    arch_IRIX=1
    if test "$CC = cc" ; then
	CFLAGS="$CFLAGS -64 -mp"
    fi
    FFLAGS="$FFLAGS -64 -mp"
fi
if test -n "$arch_LINUX" ; then
    arch_Linux=1
    if test "$CC = pgcc" ; then
	CFLAGS="$CFLAGS -mp"
    fi
    if test "$FC = pgf90" ; then
	FFLAGS="$FFLAGS -mp -Msecond_underscore"
    fi
fi
if test -n "$arch_IRIX32" ; then
    arch_IRIX=1
fi
if test -n "$arch_IRIXN32" ; then
    arch_IRIX=1
fi
if test -n "$arch_alpha" ; then
    CFLAGS="$CFLAGS -omp"
    FFLAGS="$FFLAGS -omp"
fi
#  Handle solaris on Intel platforms, needed to get heterogeneity right in p4
if test -n "$arch_solaris86" ; then
    arch_solaris=1
    ARCH=solaris86
fi
if test -n "$arch_sgi5" ; then
    arch_IRIX5=1
    ARCH=IRIX
fi
if test -n "$arch_cray" ; then
    arch_CRAY=1
    ARCH=CRAY
fi
])
