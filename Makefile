all:
	gcc mandelbrot.c -Wall -pedantic -m64 -o mandelbrot -msse3 -msse2 -msse -mmmx -mno-red-zone -momit-leaf-frame-pointer -maccumulate-outgoing-args -lgd -fast -O3 -funroll-loops -ffast-math -lpthread --std=gnu99 /usr/local/lib/libgmp.a -mdynamic-no-pic

