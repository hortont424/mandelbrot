/* 
 *    Mandelbrot Set Generator
 *    © 2008-2009, Tim Horton (hortont424@gmail.com)
 *
 *    This program renders a given portion of the Mandelbrot set to an image,
 *    and provides an example of use of the GMP, GD, and pthread libraries.
 *
 *    The aim was towards an wholly C-based, easily readable, well-commented,
 *    optimized implementation of a Mandelbrot renderer, while allowing for 
 *    parallelization and extremely-high-precision math.
 *
 *    All rights reserved.
 *
 *    Redistribution and use in source and binary forms, with or without
 *    modification, are permitted provided that the following conditions
 *    are met:
 *
 *    Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer. Redistributions in
 *    binary form must reproduce the above copyright notice, this list of
 *    conditions and the following disclaimer in the documentation and/or other
 *    materials provided with the distribution. Neither the name of the author
 *    nor the names of other contributors may be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 *    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *    POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include <gd.h>
#include <pthread.h>

#define IMAGE_WIDTH 1440    // Final output image width
#define IMAGE_HEIGHT 900    // Final output image height
#define OSA 2               // Oversampling factor applied to image
#define THREADS 10          // Number of threads to render with

#ifndef M_LN2
#define M_LN2 0.69314718055994530942
#endif

typedef struct
{
    mpf_t x;                // Mandelbrot Set horizontal center
    mpf_t y;                // Mandelbrot Set vertical center
    mpf_t r;                // Mandelbrot Set horizontal radius
    unsigned int i;         // Maximum number of iterations to calculate
} frame_info_t;

typedef struct
{
    frame_info_t frame;      // Boundary information for the current image
    unsigned int part;       // ID of the current thread
    gdImagePtr * img;        // Portion of the image rendered by this thread
} thread_info_t;

/*
 *    float mandelbrot_distance(x0, y0, max_iterations, tmp)
 *
 *    Given a point in the complex plane and the maximum number of iterations to
 *    calculate, returns a normalized iteration count based on how long it takes
 *    to exit the Mandelbrot Set.
 *    
 *    This calculation can be summed up as follows, but has been optimized
 *    beyond recognition, as it is called hundreds of thousands of times:
 * 
 *    [ z(n+1) = z(n)^2 + c ], where c is the complex point being evaluated
 *    
 *    A point is considered 'escaped' if its complex magnitude is greater than
 *    four. This number is a part of the definition of the Set itself.
 *
 *    We actually perform the complex exponentiation on our own, as GMP lacks a 
 *    complex number type):
 *        [ x(n+1) = x(n)^2 - y(n)^2 + x0 ]
 *        [ y(n+1) = 2*x(n)*y(n) + y0 ]
 *
 *    In order to smoothly color the Set, we make use of the normalized
 *    iteration count algorithm as described by Francisco Garcia, et. al., in
 *    "Coloring Dynamical Systems in the Complex Plane".
 *    
 *    [ y(n) = n - (ln(ln(abs(z(n))))/ln(2)) ]
 *
 *    This formula yields a continuously-valued count of the 'iterations' it
 *    takes to escape the Mandelbrot Set. This is directly related to the
 *    'ordinary' coloring scheme, which solely employs the iteration count and
 *    is discontinuous, and will similarly color the Set.
 *
 *    This function needs to be as fast as possible, as it is called for each
 *    oversampled pixel (so, WIDTH*HEIGHT*OSA*OSA), and loops through between
 *    one and max_iterations times. For a view of the whole Set at 1440x900x2,
 *    with 200 iterations, this results in >5000000 passes through the loop.
 *
 */
static inline float mandelbrot_distance(const mpf_t x0, const mpf_t y0,
                                        const unsigned int max_iterations,
                                        mpf_t * tmp)
{
    mpf_set(tmp[2], x0);
    mpf_set(tmp[3], y0);
    
    unsigned int extra_iterations = 0;
    
    for(unsigned int iteration = max_iterations;
        iteration > 0 || extra_iterations; --iteration)
    {
        mpf_mul(tmp[0], tmp[2], tmp[2]);
        mpf_mul(tmp[1], tmp[3], tmp[3]);
        
        mpf_add(tmp[0], tmp[0], tmp[1]);
        
        // Check Mandelbrot distance (x^2 + y^2 <= 4).
        // Iterate 4 extra times if we're exiting, to smooth coloring.
        if(extra_iterations || mpf_cmp_ui(tmp[0], 4) >= 0)
            if(++extra_iterations == 4)
                return (max_iterations - iteration) - 
                       (logf(logf(mpf_get_d(tmp[0])))/M_LN2);
            
        // Do the first subtraction, also fix the addition above.
        mpf_mul_2exp(tmp[1], tmp[1], 1);
        mpf_sub(tmp[0], tmp[0], tmp[1]);
        
        mpf_mul(tmp[1], tmp[2], tmp[3]);
        mpf_mul_2exp(tmp[1], tmp[1], 1);
        
        mpf_add(tmp[3], tmp[1], y0);
        mpf_add(tmp[2], tmp[0], x0);
    }

    return 0;
}

/*
 *    int colorize(distance, max_iterations)
 *
 *    Choose a color, based on the normalized iteration count.
 *    Different coloring methods could be used.
 *
 */
static inline int colorize(float distance, unsigned int max_iterations)
{
    return gdTrueColor((unsigned int)(distance/max_iterations*255),
                       (unsigned int)((cosf(0.10*distance)+1)*127),
                       (unsigned int)((sinf(0.01*distance)+1)*127));
}

/*
 *    void * render_thread(thread_info_in)
 *
 *    The main body of each rendering thread. Given a thread_info_t, this
 *    function iterates through the image, calculating a color for each pixel.
 * 
 *    The image, stored in thread_info_t.img, represents a portion of the
 *    requested image: the full width of the output image, and 1/THREADS high.
 *
 */
static void * render_thread(void * thread_info_in)
{
    thread_info_t * thread_info = (thread_info_t *)thread_info_in;
    
    // Scale the radius based on the image's aspect ratio and the thread count
    mpf_t mandelbrot_aspect, mandelbrot_rscaled,
          mandelbrot_rscaledproc, partno;
    
    mpf_init(mandelbrot_rscaled);
    mpf_init_set_d(mandelbrot_aspect, ((float)IMAGE_WIDTH) / IMAGE_HEIGHT);
    mpf_div(mandelbrot_rscaled, thread_info->frame.r, mandelbrot_aspect);
    mpf_init_set(mandelbrot_rscaledproc, mandelbrot_rscaled);
    mpf_init_set_d(partno, 2*((float)thread_info->part)/THREADS);
    mpf_mul(mandelbrot_rscaledproc, mandelbrot_rscaledproc, partno);
    
    // Calculate minimum and maximum Y locations based on the thread count
    unsigned int min_y = ((float)thread_info->part)/THREADS*IMAGE_HEIGHT*OSA;
    unsigned int max_y = ((float)thread_info->part+1)/THREADS*IMAGE_HEIGHT*OSA;
    
    // Initialize four temporary variables for the inner loop
    mpf_t tmp[4]; for(unsigned int i = 0; i < 4; ++i) mpf_init(tmp[i]);
    
    // Calculate the complex X location of the left side of the image
    mpf_t mandelbrot_min_x;
    mpf_init_set(mandelbrot_min_x, thread_info->frame.x);
    mpf_sub(mandelbrot_min_x, mandelbrot_min_x, thread_info->frame.r);
    
    // Calculate the relationship between the image's coordinates and the
    // complex plane. This 'step' factor is added to X and Y each time around
    // their respective loops.
    mpf_t mandelbrot_step_x, mandelbrot_step_y;
    mpf_init(mandelbrot_step_x);
    mpf_init(mandelbrot_step_y);
    mpf_mul_2exp(mandelbrot_step_x, thread_info->frame.r, 1);
    mpf_mul_2exp(mandelbrot_step_y, mandelbrot_rscaled, 1);
    mpf_div_ui(mandelbrot_step_x, mandelbrot_step_x, IMAGE_WIDTH*OSA);
    mpf_div_ui(mandelbrot_step_y, mandelbrot_step_y, IMAGE_HEIGHT*OSA);
    
    // Calculate the initial complex Y location
    mpf_t mandelbrot_x, mandelbrot_y;
    mpf_init(mandelbrot_x);
    mpf_init_set(mandelbrot_y, thread_info->frame.y);
    mpf_sub(mandelbrot_y, mandelbrot_y, mandelbrot_rscaled);
    mpf_add(mandelbrot_y, mandelbrot_y, mandelbrot_rscaledproc);
    
    float xreal, yreal, dist;
    unsigned int image_x, image_y;
    
    // Step through both the image's coordinate plane and the complex plane.
    // Draw each pixel into the GD image in memory, individually.
    // It might be faster to write to an array and then copy to GD.
    for(image_y = min_y; image_y < max_y;
        ++image_y, mpf_add(mandelbrot_y, mandelbrot_y, mandelbrot_step_y))
    {
        mpf_set(mandelbrot_x, mandelbrot_min_x);
        
        yreal = mpf_get_d(mandelbrot_y);
        yreal *= yreal;
        
        for(image_x = 0; image_x < (IMAGE_WIDTH*OSA);
            ++image_x, mpf_add(mandelbrot_x, mandelbrot_x, mandelbrot_step_x))
        {
            xreal = mpf_get_d(mandelbrot_x);
            
            // Check that we're not inside one of the main bulbs or the cardioid
            if((xreal+1)*(xreal+1) + yreal < .0625 ||
               (xreal+1.310)*(xreal+1.310) + yreal < .0036 ||
               (xreal+1.381)*(xreal+1.381) + yreal < .00017 ||
               (xreal-.25)*(xreal-.25)+yreal < 
                    (float)(1-cosf(atan2f(mpf_get_d(mandelbrot_y), xreal)))/2)
                continue;

            // Smoothly colorize pixels based on the number of iterations
            // it takes to escape the set.
            dist = mandelbrot_distance(mandelbrot_x, mandelbrot_y,
                                       thread_info->frame.i, tmp);
            if(dist != 0.0)
                gdImageSetPixel(*(thread_info->img), image_x, image_y-min_y,
                                colorize(dist, thread_info->frame.i));
        }
    }
    
    // Free all of our temporary variables
    for(unsigned int i = 0; i < 4; ++i) mpf_clear(tmp[i]);
    mpf_clear(mandelbrot_x);
    mpf_clear(mandelbrot_y);
    mpf_clear(mandelbrot_rscaled);
    mpf_clear(mandelbrot_rscaledproc);
    mpf_clear(partno);
    mpf_clear(mandelbrot_aspect);
    
    pthread_exit(NULL);
}

void throw_error(const char * err_str, const int err_no)
{
    fprintf(stderr, "Error %i: %s. Render terminated.", err_no, err_str);
    exit(err_no);
}

/*
 *    void render_frame(frame, filename)
 *
 *    Given a frame_info_t structure with the properties of the requested Set,
 *    and a filename in which to save the rendered image, spins off THREADS
 *    running threads, each one running in parallel, running vertically divided
 *    portions of the requested Set.
 *
 *    When rendering is complete, scale the oversampled parts of the image and
 *    copy them down into an image of the final output size. Then, write the 
 *    image, as a PNG, to the given filename, with no compression.
 *
 */
void render_frame(const frame_info_t frame, const char * filename)
{
    pthread_t threads[THREADS];
    thread_info_t * tinfo[THREADS];

    // Each thread needs to be joinable so we can wait for all threads to finish
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    // Create and start THREADS threads, with needed data, including a pointer
    // to a new, unique gdImagePtr for each thread.
    for(unsigned int t = 0; t < THREADS; ++t)
    {
        tinfo[t] = (thread_info_t *)malloc(sizeof(thread_info_t));
        
        tinfo[t]->img = (gdImagePtr *)malloc(sizeof(gdImagePtr *));
        *(tinfo[t]->img) = gdImageCreateTrueColor((IMAGE_WIDTH*OSA),
                                                  (IMAGE_HEIGHT*OSA)/THREADS);
        tinfo[t]->part = t;
        tinfo[t]->frame = frame;
        
        if(pthread_create(&threads[t], &attr, render_thread, (void *)tinfo[t]))
            throw_error("Error Creating Thread", -1);
    }
    
    pthread_attr_destroy(&attr);
    
    // Create the output image, of required size, and resample and copy each
    // of the threads' images into it. This resampling smooths over the colors,
    // and is especially important when creating animations of the Set.
    gdImagePtr output_image = gdImageCreateTrueColor(IMAGE_WIDTH, IMAGE_HEIGHT);
    for(unsigned int t = 0; t < THREADS; ++t)
    {
        if(pthread_join(threads[t], (void **)NULL))
            throw_error("Error Joining Thread", -2);
        else
        {
            gdImageCopyResampled(output_image,
                                 *(tinfo[t]->img),
                                 0, (((float)t)/THREADS)*IMAGE_HEIGHT,
                                 0, 0,
                                 IMAGE_WIDTH, IMAGE_HEIGHT/THREADS,
                                 (IMAGE_WIDTH*OSA), (IMAGE_HEIGHT*OSA)/THREADS);
            gdImageDestroy(*(tinfo[t]->img));
            free(tinfo[t]->img);
            free(tinfo[t]);
        }
    }
    
    // Write the final image out to a PNG file.
    FILE * pngout = fopen(filename, "wb");
    gdImagePngEx(output_image, pngout, 0);
    gdImageDestroy(output_image);
    fclose(pngout);
} 

int main(int argc, char const ** argv)
{
    frame_info_t currentFrame;
    
    mpf_init_set_str(currentFrame.x, "-0.75@0", 10);
    mpf_init_set_str(currentFrame.y, "0.00@0", 10);
    mpf_init_set_str(currentFrame.r, "1.80@0", 10);

    currentFrame.i = 200;
    
    render_frame(currentFrame, "mandelbrot.png");
    
    mpf_clear(currentFrame.x);
    mpf_clear(currentFrame.y);
    mpf_clear(currentFrame.r);
        
    return 0;
}