#ifndef _PREPROCESSOR_HPP
#define _PREPROCESSOR_HPP

#define TO_STRING_INTERNAL(x) #x
#define TO_STRING(x) TO_STRING_INTERNAL(x)

#define LINE_AS_STRING TO_STRING(__LINE__)

#define HERE __FILE__ ":" LINE_AS_AS_STRING

#endif
