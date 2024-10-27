#include "autodifferentation.h"
#include "raylib.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

float calc_pos_delta(int frag, int total) {
    return (float) total/(frag + 1);
}

typedef struct memorizer {
    int size;
    int capacity;
    ADNode** start;
} memorizer;

int _depth(ADNode* head, int n) {
    int max = n;
    for(int i = 0; i < head->num_parents; i++) {
        int mn = _depth(head->parents[i], n+1);
        max = max > mn ? max : mn;
    }
    return max;
}

memorizer* _init_memorizer() {
    memorizer* m = (memorizer*) MemAlloc(sizeof(memorizer));
    m->size = 0;
    m->capacity = 10;
    m->start = (ADNode**) MemAlloc(sizeof(ADNode*)*10);
    return m;
}
char _contains(memorizer* m, ADNode* elem) {
    for(int i = 0; i < m->size; i++) {
        if(m->start[i] == elem) return 1;
    }
    return 0;;
}
void _fill_depth(ADNode* r, int n) {
    r->depth = r->depth < n ? r->depth : n;
    for(int i = 0; i < r->num_parents; i++) {
        _fill_depth(r->parents[i], r->depth + 1);
    }
}
void _flush(ADNode* r) {
    if(r->depth == INT_MAX) {
        return;
    }
    r->depth = INT_MAX;
    for(int i = 0; i < r->num_parents; i++) {
        _flush(r->parents[i]);
    }
}
void _append(memorizer* to_fill, ADNode* elem) {
    if(_contains(to_fill, elem)) return;
    if(to_fill->capacity == to_fill->size) {
        to_fill->start = (ADNode**) MemRealloc((void*) to_fill->start,sizeof(ADNode*) * to_fill->capacity * 2);
        to_fill->capacity *= 2;
    }
    to_fill->start[to_fill->size++] = elem;
}
void _fill_node(memorizer* to_fill, int n, ADNode* r) {
    if(r->depth == n) {
        _append(to_fill, r);
    }
    for(int i = 0; i < r->num_parents; i++) {
        _fill_node(to_fill, n, r->parents[i]);
    }
}

void _calc_vec(Vector2* p, int n, float x, float delta_y) {
    for(int i = 0; i < n; i++) {
        p[i].x = x;
        p[i].y = delta_y * (i+1);
    }
}
int _index_of(memorizer* m, ADNode* el) {
    for(int i = 0; i < m->size; i++) {
        if(m->start[i] == el) {
            return i;
        }
    }
    return -1;
}

void draw_graph(ADNode* head, int window_h, int window_w) {
    _flush(head);
    _fill_depth(head,0);
    int depth = _depth(head, 1);
    memorizer** mem = (memorizer**) MemAlloc(sizeof(memorizer*)*depth);
    for(int i = 0; i < depth; i++) {
        mem[i] = _init_memorizer();
        _fill_node(mem[i],i,head);
    }
    float delta_x = calc_pos_delta(depth, window_w);
    Vector2** pos = (Vector2**) MemAlloc(sizeof(Vector2*)*depth);
    for(int i = 0; i < depth; i++) {
        pos[i] = (Vector2*) MemAlloc(sizeof(Vector2)*mem[i]->size);
        float x = window_w - (delta_x * (i+1));
        float delta_y = calc_pos_delta(mem[i]->size, window_h);
        _calc_vec(pos[i], mem[i]->size, x, delta_y);
    }

    InitWindow(window_w*2, window_h*2, "Network Visual");
    SetTargetFPS(1);
    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(RAYWHITE);
        DrawLineV((Vector2){window_w,0}, (Vector2){window_w,window_h}, BLACK);
        DrawLineV((Vector2){0,window_h}, (Vector2){window_w,window_h}, BLACK);

        for(int i = depth - 1; i>= 0; i--) {
            for(int j = 0; j < mem[i]->size; j++) {
                DrawCircleV(pos[i][j], 10, BLACK);
                for(int k = 0; k < (mem[i]->start)[j]->num_parents;k++) {
                    int dot = _index_of(mem[i+1], (mem[i]->start)[j]->parents[k]);
                    DrawLineV(pos[i+1][dot], pos[i][j], BLACK);
                }
            }
        }
        EndDrawing();
    }
    CloseWindow();
    for(int i = 0; i < depth; i++) {
        MemFree(pos[i]);
        MemFree(mem[i]);
    }
    MemFree(pos);
    MemFree(mem);
}
