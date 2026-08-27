#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <xstatus.h>
#include <xparameters.h>
#include <xil_printf.h>
#include <string.h>
#include <xiltimer.h>
#include <xtimer_config.h>
#include <sleep.h>

#include "token_id.h"


#define VOCAB_SIZE      3005
#define POEM_LEN        56
#define TITLE_LEN       10
#define TOKEN_ID_BOS    0
#define TOKEN_ID_SEP    2

volatile unsigned int *template_id_addr = (unsigned int *) (XPAR_ENGINE_WRAPPER_0_BASEADDR);
volatile unsigned int *title_base_addr  = (unsigned int *) ((XPAR_ENGINE_WRAPPER_0_BASEADDR) + (0x04));
volatile unsigned int *title_len_addr   = (unsigned int *) ((XPAR_ENGINE_WRAPPER_0_BASEADDR) + (0x34));
volatile unsigned int *lcg_seed_addr    = (unsigned int *) ((XPAR_ENGINE_WRAPPER_0_BASEADDR) + 0x38);
volatile unsigned int *poem_start_addr  = (unsigned int *) (XPAR_ENGINE_WRAPPER_0_BASEADDR + 0x3C);
volatile unsigned int *poem_end_addr    = (unsigned int *) (XPAR_ENGINE_WRAPPER_0_BASEADDR + 0x40);
volatile unsigned int *poem_base_addr   = (unsigned int *) (XPAR_ENGINE_WRAPPER_0_BASEADDR + 0x44);

unsigned int title_arr[TITLE_LEN];
unsigned int poem_arr[POEM_LEN];

char title_str[] = "送别";

static int utf8_len(unsigned char c) {
    if (c < 0x80)           return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return -1;
}

int find_token_id (char *title_str) {
    for (size_t i = 0; i < VOCAB_SIZE; i++) {
        if (strcmp(token_id_map[i], title_str) == 0) {
            return i;
        }
    }
    return -1;
}

int encode_title(char *title_str) {
    size_t byte_pos = 0;
    size_t n_chars  = 0;
    
    while (title_str[byte_pos] != '\0') {
        int len_char = utf8_len((unsigned char) title_str[byte_pos]);
        
        if (n_chars > TITLE_LEN) {
            xil_printf("[Warning] Title too long! Must be within %d characters.\r\n", TITLE_LEN);
            return -1;
        }
        
        char single_char[5] = {0};
        for (int j = 0; j < len_char; j++) {
            single_char[j] = title_str[byte_pos + j];
        }
        
        int token_id = find_token_id(single_char);
        if (token_id == -1) {
            xil_printf("[Warning] Character %s is not in vocabulary!\r\n", single_char);
            return -1;
        }
        
        title_arr[n_chars] = token_id;
        n_chars++;
        byte_pos += len_char;
    }
    
    return (int) n_chars;
}

int assign_template (unsigned int template_id) {
    if (template_id >= 1 && template_id <= 4) {
        *template_id_addr = template_id - 1;
        return XST_SUCCESS;
    } else {
        xil_printf("[Warning] Template id out of range (1-4)!\r\n");
        return XST_FAILURE;
    }
}

int assign_title (unsigned int *title, size_t size) {
    volatile unsigned int *title_addr;
    
    if (size > TITLE_LEN) {
        xil_printf("[Warning] Title too long! Must be within %d characters.\r\n", TITLE_LEN);
        return XST_FAILURE;
    }

    *title_base_addr = TOKEN_ID_BOS;
    
    for (size_t i = 0; i < size; i++) {
        if (title[i] >= VOCAB_SIZE) {
            xil_printf("[Warning] Character %d not in vocabulary!\r\n", title[i]);
            return XST_FAILURE;
        }
        title_addr = title_base_addr + (i + 1);
        *title_addr = title[i];
    }
    
    *title_len_addr = size + 2;

    title_addr = title_base_addr + (size + 1);
    *title_addr = TOKEN_ID_SEP;
    return XST_SUCCESS;
}

int assign_lcg_seed (uint32_t seed) {
    *lcg_seed_addr = seed;
    return XST_SUCCESS;
}

int start_poem() {
    *poem_start_addr = 1;
    return XST_SUCCESS;
}

int check_poem_done() {
    int is_poem_done = *poem_end_addr;
    return is_poem_done;
}

int read_poem() {
    volatile unsigned int *poem_addr; 

    for (size_t i = 0; i < POEM_LEN; i++) {
        poem_addr = poem_base_addr + i;
        poem_arr[i] = *poem_addr;
    }
    
    return XST_SUCCESS;
}

int print_poem() {
    for (size_t i = 0; i < POEM_LEN; i++) {
        xil_printf("%s", token_id_map[poem_arr[i]]);
        usleep(10000); // delay to prevent UART overflow
        if (i % 14 == 6) {
            xil_printf("，");
        } else if (i % 14 == 13) {
            xil_printf("。\r\n");
        }
    }

    return XST_SUCCESS;
}

int main () {
    
    XTime t_start, t_end;
    
    int size = encode_title(title_str);

    if (size == -1) {
        return XST_FAILURE;
    }
    
    assign_template(1);
    assign_title(title_arr, size);
    assign_lcg_seed(20260815);
    
    start_poem();
    XTime_GetTime(&t_start);

    while (!check_poem_done()) {
        // wait for poem to finish
    }
    
    XTime_GetTime(&t_end);

    read_poem();
    print_poem();

    XTime elapsed_tics = t_end - t_start;
    u32 elapsed_us = (u32)(((u64)elapsed_tics * 1000000ULL) / (u64)(COUNTS_PER_SECOND));
    xil_printf("Generation time: %d.%03d ms\r\n", elapsed_us / 1000, elapsed_us % 1000);
    
    return XST_SUCCESS;
    
}