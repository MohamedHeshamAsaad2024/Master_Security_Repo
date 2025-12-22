#include <iostream>
#include <vector>
#include <string>

std::vector<uint8_t> SHA_MultiRatePadding(const std::string& input, int rate) 
{
    /* Convert the input string to a vector of bytes */
    std::vector<uint8_t> paddedMessage(input.begin(), input.end());

    /* Determine the rate in bytes */
    size_t rateInBytes = rate / 8;

    /* Compute the length of the input in bytes */
    size_t m = paddedMessage.size();

    /* Calculate the number of padding bytes (q) */
    size_t q = rateInBytes - (m % rateInBytes);

    /* Append the padding bytes depending on the needed number of padding bytes (q) */
    if (q == 1) 
    {
        /* If the input message is multiple of the rate, then we only append one byte */
        paddedMessage.push_back(0x86);
    } 
    else if (q == 2) 
    {
        /* If two bytes are needed, we set the first byte 0x06 as the prefix byte */
        paddedMessage.push_back(0x06);
        /* We add the last byte which terminates the padding */
        paddedMessage.push_back(0x80);
    } else 
    {
        /* If more than 2 bytes are needed, we set the first byte 0x06 as the prefix byte */
        paddedMessage.push_back(0x06);

        /* Add intermediate zero bytes */
        for (size_t i = 0; i < q - 2; ++i) 
        {
            paddedMessage.push_back(0x00);
        }

        /* We add the last byte which terminates the padding */
        paddedMessage.push_back(0x80);
    }

    return paddedMessage;
}

int main() {
    std::string message = "Hello, SHA-3!";
    size_t rate = 1088; // Example rate (in bits)

    // Perform padding
    std::vector<uint8_t> paddedMessage = SHA_MultiRatePadding(message, rate);

    // Output the padded message as hex
    std::cout << "Padded Message: ";
    for (uint8_t byte : paddedMessage) {
        std::cout << std::hex << static_cast<int>(byte) << " ";
    }
    std::cout << std::endl;

    return 0;
}
