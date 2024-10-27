#include "visualizer.h"
#include "autodifferentation.h"
#include <stdlib.h>

int main(void) {
    ADNode* aa = (ADNode*)malloc(sizeof(ADNode));
    ADNode* b = (ADNode*)malloc(sizeof(ADNode));
    ADNode* c = (ADNode*)malloc(sizeof(ADNode));
    ADNode* d = (ADNode*)malloc(sizeof(ADNode));
    ADNode* e = (ADNode*)malloc(sizeof(ADNode));
    ADNode* g = (ADNode*)malloc(sizeof(ADNode));

    aa->parents = (ADNode**)malloc(sizeof(ADNode*) * 2);
    aa->num_parents = 2;
    aa->parents[0] = b;
    aa->parents[1] = c;
    b->parents = (ADNode**)malloc(sizeof(ADNode*)*3);
    b->parents[0] = d;
    b->parents[1] = e;
    b->parents[2] = g;
    c->parents = (ADNode**)malloc(sizeof(ADNode*)*3);

    c->parents[0] = d;
    c->parents[1] = e;
    c->parents[2] = g;
    b->num_parents = 3;
    c->num_parents = 3;

    draw_graph(aa,400,400);
    free(aa->parents);
    free(b->parents);
    free(c->parents);
    free(aa);
    free(b);
    free(c);
    free(d);
    return 0;
}
