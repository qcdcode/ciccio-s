
# usage: AX_GENERATE_SIDE_ASSEMBLY
AC_DEFUN([AX_GENERATE_SIDE_ASSEMBLY], [

AX_GET_ENABLE(generate_side_assembly,false,"Automatic generate assembly")

AM_CONDITIONAL([GENERATE_SIDE_ASSEMBLY],[test "$enable_generate_side_assembly" != "false" ],[true],[false])
])
