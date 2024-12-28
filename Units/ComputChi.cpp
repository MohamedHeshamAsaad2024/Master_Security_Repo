#include <cstdint>
#include <cstdio>
#include <iostream>

#define STATE_ROW_SIZE              5U
#define STATE_COLUMN_SIZE           5U

void SHA_ComputeChi(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE]) 
{
    /* Define a temporary state to hold the original states before computation */
    uint64_t tempState[STATE_ROW_SIZE][STATE_COLUMN_SIZE];

    /* Copy the current state into the temporary state */
    for (int x = 0; x < STATE_ROW_SIZE; ++x) 
    {
        for (int y = 0; y < STATE_COLUMN_SIZE; ++y) 
        {
            tempState[x][y] = state[x][y];
        }
    }

    /* Apply the Chi step mapping */
    for (int x = 0; x < STATE_ROW_SIZE; ++x) 
    {
        for (int y = 0; y < STATE_COLUMN_SIZE; ++y) 
        {
            state[x][y] = tempState[x][y] ^ 
                          ((~tempState[(x + 1) % STATE_ROW_SIZE][y]) & 
                           tempState[(x + 2) % STATE_ROW_SIZE][y]);
        }
    }
}

void printState(uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE])
{
    int byteCount = 0; // Counter for bytes printed

    /* Print the updated state */
    for (int x = 0; x < STATE_ROW_SIZE; ++x)
    {
        for (int y = 0; y < STATE_COLUMN_SIZE; ++y)
        {
            uint8_t *bytes = reinterpret_cast<uint8_t*>(&state[x][y]);
            for (int i = 0; i < sizeof(uint64_t); ++i)
            {
                printf("%02X ", bytes[i]);
                byteCount++;

                if (byteCount % 16 == 0) { // Newline after every 16 bytes
                    printf("\n");
                }
            }
        }
    }
    printf("\n"); // Final newline
}


int main() 
{
    uint64_t state[STATE_ROW_SIZE][STATE_COLUMN_SIZE] = 
    {
        {0x00000001997B5852, 0xB585300000001997, 0x0000040000000000, 0x0000000000000000, 0x0000CCBDAC298000},
        {0x0000000000000000, 0x00332F6B0A600000, 0x0000000000000008, 0x6B0A70000000332F, 0x1000000000000000},
        {0x0000000332F6B0A6, 0x0000000000000020, 0x0000000000000000, 0x00000332F6B0A600, 0x0000000000040000},
        {0x1997B58530000000, 0x0000001000000000, 0x00000665ED614C00, 0x0000000000004000, 0x0000000000000000},
        {0x2000000000000000, 0x0000000000000000, 0x7B58530000000199, 0x0000020000000000, 0x0000000665ED614C}
    };

    std::cout << "Input State:" << std::endl;
    printState(state);

    SHA_ComputeChi(state);

    std::cout << "After Chi Step:" << std::endl;
    printState(state);

    return 0;
}