#include <cstdio>
#include <cstdlib>
#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "segment-image.h"

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s sigma k min input(ppm) canny(ppm) output(ppm)\n", argv[0]);
        return 1;
    }

    float sigma = atof(argv[1]);
    float k = atof(argv[2]);
    int min_size = atoi(argv[3]);

    image<rgb> *input = loadPPM(argv[4]);
    image<rgb> *EDGE = loadPPM(argv[5]);
    int width = input->width();
    int height = input->height();
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if(EDGE->access[y][x].r == 255) {
                input->access[y][x].r = input->access[y][x].g = input->access[y][x].b = 256;
            }
        }
    }

    printf("Cutting graph...\n");
    int num_ccs;
    image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs);
    savePPM(seg, argv[6]);

    printf("got %d components.\n", num_ccs);
    return 0;
}

