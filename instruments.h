#ifndef _INSTRUMENTS_H
#define _INSTRUMENTS_H

#if USE_OSACA
#define BEGIN_INSTRUMENTATION BEGIN_OSACA
#define END_INSTRUMENTATION   END_OSACA
#elif USE_MCA
#define BEGIN_INSTRUMENTATION BEGIN_MCA
#define END_INSTRUMENTATION   END_MCA
#else
#define BEGIN_INSTRUMENTATION
#define END_INSTRUMENTATION
#endif



#define BEGIN_MCA __asm volatile("# LLVM-MCA-BEGIN foo":::"memory");
#define END_MCA   __asm volatile("# LLVM-MCA-END":::"memory");


#define BEGIN_OSACA __asm volatile("\n\t movl    $111,%%ebx #IACA/OSACA START MARKER" \
                                   "\n\t .byte   100,103,144     #IACA/OSACA START MARKER" \
				   :::"memory");

#define END_OSACA __asm volatile("\n\t movl    $222,%%ebx #IACA/OSACA START MARKER" \
				 "\n\t .byte   100,103,144     #IACA/OSACA START MARKER" \
				   :::"memory");


#endif /* _INSTRUMENTS_H */
