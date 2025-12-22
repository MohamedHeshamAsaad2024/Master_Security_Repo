#include <iostream>
#include <gmpxx.h> // GMP library for handling large integers
#include <vector>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For seeding rand()
using namespace std;

/**
 * Function to pad an ordinary message in PKCS#1 v1.5 format.
 * Converts the plaintext message into a padded message suitable for RSA encryption.
 * 
 * @param PaddedMessage: Reference to the resulting padded message as an mpz_class.
 * @param OrdinaryMessage: The original plaintext message as an mpz_class.
 */
void ConvertOrdinaryMessageToPaddedMessage(mpz_class &PaddedMessage, const mpz_class OrdinaryMessage) {
    // Define key size in bytes (e.g., 256 bytes for 2048-bit RSA)
    size_t keySize = 256;

    // Convert the plaintext to a byte array
    vector<uint8_t> plaintext = mpzToByteArray(OrdinaryMessage, OrdinaryMessage.get_str(16).size() / 2);

    // Ensure the plaintext fits within the padded structure
    if (plaintext.size() > keySize - 11) {
        throw invalid_argument("Plaintext is too long for the given key size.");
    }

    // Initialize the padded message
    vector<uint8_t> paddedMessage(keySize, 0x00);

    // Add padding format bytes
    paddedMessage[0] = 0x00;
    paddedMessage[1] = 0x02;

    // Add random non-zero padding
    size_t paddingLength = keySize - plaintext.size() - 3;
    srand(static_cast<unsigned>(time(0)));
    for (size_t i = 0; i < paddingLength; ++i) {
        uint8_t randomByte;
        do {
            randomByte = rand() % 256;
        } while (randomByte == 0x00);
        paddedMessage[2 + i] = randomByte;
    }

    // Add the separator and plaintext
    paddedMessage[2 + paddingLength] = 0x00;
    memcpy(&paddedMessage[3 + paddingLength], plaintext.data(), plaintext.size());

    // Convert padded message to mpz_class
    PaddedMessage = byteArrayToMpz(paddedMessage);
}


/**
 * Function to remove PKCS#1 v1.5 padding from a padded message.
 * Converts a padded message back into the original plaintext message.
 * 
 * @param OrdinaryMessage: Reference to the resulting plaintext message as an mpz_class.
 * @param PaddedMessage: The padded message as an mpz_class.
 */
void ConvertPaddedMessageToOrdinaryMessage(mpz_class &OrdinaryMessage, const mpz_class PaddedMessage) {
    // Define key size in bytes
    size_t keySize = 256;

    // Convert padded message to a byte array
    vector<uint8_t> paddedBytes = mpzToByteArray(PaddedMessage, keySize);

    // Verify padding format
    if (paddedBytes[0] != 0x00 || paddedBytes[1] != 0x02) {
        throw invalid_argument("Invalid padding format.");
    }

    // Find the separator
    size_t separatorIndex = 2;
    while (separatorIndex < paddedBytes.size() && paddedBytes[separatorIndex] != 0x00) {
        separatorIndex++;
    }

    if (separatorIndex >= paddedBytes.size()) {
        throw invalid_argument("Padding separator not found.");
    }

    // Extract the plaintext
    vector<uint8_t> plaintext(paddedBytes.begin() + separatorIndex + 1, paddedBytes.end());
    OrdinaryMessage = byteArrayToMpz(plaintext);
}

int main() {
    // Example usage
    mpz_class OrdinaryMessage("48656C6C6F", 16); // "Hello" in hexadecimal
    mpz_class PaddedMessage;

    // Convert to padded message
    ConvertOrdinaryMessageToPaddedMessage(PaddedMessage, OrdinaryMessage);
    cout << "Padded Message: " << PaddedMessage.get_str(16) << endl;

    // Convert back to ordinary message
    mpz_class RecoveredMessage;
    ConvertPaddedMessageToOrdinaryMessage(RecoveredMessage, PaddedMessage);
    cout << "Recovered Message: " << RecoveredMessage.get_str(16) << endl;

    return 0;
}


vector<uint8_t> mpzToByteArray(const mpz_class &num, size_t byteLength) {
    // Convert the number to a hexadecimal string
    string hexStr = num.get_str(16);

    // Calculate the number of leading zeros required
    size_t hexLength = byteLength * 2; // Each byte is 2 hex characters
    if (hexStr.size() < hexLength) {
        hexStr = string(hexLength - hexStr.size(), '0') + hexStr;
    }

    // Convert the hex string to a byte array
    vector<uint8_t> byteArray(byteLength);
    for (size_t i = 0; i < byteLength; ++i) {
        byteArray[i] = stoi(hexStr.substr(2 * i, 2), nullptr, 16);
    }

    return byteArray;
}

mpz_class byteArrayToMpz(const vector<uint8_t> &byteArray) {
    string hexStr;
    for (uint8_t byte : byteArray) {
        char buf[3];
        sprintf(buf, "%02X", byte); // Ensure 2-character hex representation
        hexStr += buf;
    }
    return mpz_class(hexStr, 16); // Convert from hex string to mpz_class
}

void ConvertOrdinaryMessageToPaddedMessage(mpz_class &PaddedMessage, const mpz_class OrdinaryMessage) {
    // Define key size in bytes (e.g., 256 bytes for 2048-bit RSA)
    size_t keySize = 256;

    // Convert the plaintext to a byte array
    vector<uint8_t> plaintext = mpzToByteArray(OrdinaryMessage, OrdinaryMessage.get_str(16).size() / 2);

    // Ensure the plaintext fits within the padded structure
    if (plaintext.size() > keySize - 11) {
        // Split the message into blocks if it's too large
        size_t blockSize = keySize - 11;
        size_t numBlocks = (plaintext.size() + blockSize - 1) / blockSize;  // Round up to get number of blocks

        vector<uint8_t> paddedMessage;
        
        // Process each block
        for (size_t i = 0; i < numBlocks; ++i) {
            size_t blockStart = i * blockSize;
            size_t blockEnd = min(blockStart + blockSize, plaintext.size());
            vector<uint8_t> block(plaintext.begin() + blockStart, plaintext.begin() + blockEnd);
            
            // Initialize the padded block vector
            vector<uint8_t> paddedBlock(keySize, 0x00);
            paddedBlock[0] = 0x00;
            paddedBlock[1] = 0x02;

            // Add random non-zero padding
            size_t paddingLength = keySize - block.size() - 3;
            srand(static_cast<unsigned>(time(0)));
            for (size_t j = 0; j < paddingLength; ++j) {
                uint8_t randomByte;
                do {
                    randomByte = rand() % 256;
                } while (randomByte == 0x00);  // Ensure non-zero padding
                paddedBlock[2 + j] = randomByte;
            }

            // Add the separator and the block data
            paddedBlock[2 + paddingLength] = 0x00;
            memcpy(&paddedBlock[3 + paddingLength], block.data(), block.size());

            // Append the padded block to the final padded message
            paddedMessage.insert(paddedMessage.end(), paddedBlock.begin(), paddedBlock.end());
        }

        // Convert the concatenated padded message to mpz_class
        PaddedMessage = byteArrayToMpz(paddedMessage);
    } else {
        // If the message is small enough to fit in one block, apply padding normally
        vector<uint8_t> paddedMessage(keySize, 0x00);
        paddedMessage[0] = 0x00;
        paddedMessage[1] = 0x02;

        // Add random non-zero padding
        size_t paddingLength = keySize - plaintext.size() - 3;
        srand(static_cast<unsigned>(time(0)));
        for (size_t i = 0; i < paddingLength; ++i) {
            uint8_t randomByte;
            do {
                randomByte = rand() % 256;
            } while (randomByte == 0x00);  // Ensure non-zero padding
            paddedMessage[2 + i] = randomByte;
        }

        // Add the separator and plaintext
        paddedMessage[2 + paddingLength] = 0x00;
        memcpy(&paddedMessage[3 + paddingLength], plaintext.data(), plaintext.size());

        // Convert the padded message to mpz_class
        PaddedMessage = byteArrayToMpz(paddedMessage);
    }
}

/*********************************************************************************************************************************************************************************************************************** */

void ConvertOrdinaryMessageToPaddedMessage(mpz_class &PaddedMessage, const mpz_class OrdinaryMessage) {
    // Define key size in bytes (e.g., 256 bytes for 2048-bit RSA)
    size_t keySize = 256;

    // Convert the plaintext to a byte array
    vector<uint8_t> plaintext = mpzToByteArray(OrdinaryMessage, OrdinaryMessage.get_str(16).size() / 2);

    // Ensure the plaintext fits within the padded structure
    if (plaintext.size() > keySize - 11) {
        // Split the message into blocks if it's too large
        size_t blockSize = keySize - 11;
        size_t numBlocks = (plaintext.size() + blockSize - 1) / blockSize;  // Round up to get number of blocks

        vector<uint8_t> paddedMessage;

        // Process each block
        for (size_t i = 0; i < numBlocks; ++i) {
            size_t blockStart = i * blockSize;
            size_t blockEnd = min(blockStart + blockSize, plaintext.size());
            vector<uint8_t> block(plaintext.begin() + blockStart, plaintext.begin() + blockEnd);
            
            // Initialize the padded block vector
            vector<uint8_t> paddedBlock(keySize, 0x00);
            paddedBlock[0] = 0x00;
            paddedBlock[1] = 0x02;

            // Add random non-zero padding
            size_t paddingLength = keySize - block.size() - 3;
            srand(static_cast<unsigned>(time(0)));
            for (size_t j = 0; j < paddingLength; ++j) {
                uint8_t randomByte;
                do {
                    randomByte = rand() % 256;
                } while (randomByte == 0x00);  // Ensure non-zero padding
                paddedBlock[2 + j] = randomByte;
            }

            // Add the separator and the block data
            paddedBlock[2 + paddingLength] = 0x00;
            memcpy(&paddedBlock[3 + paddingLength], block.data(), block.size());

            // Append the padded block to the final padded message
            paddedMessage.insert(paddedMessage.end(), paddedBlock.begin(), paddedBlock.end());
        }

        // Convert the concatenated padded message to mpz_class
        PaddedMessage = byteArrayToMpz(paddedMessage);
    } else {
        // If the message is small enough to fit in one block, apply padding normally
        vector<uint8_t> paddedMessage(keySize, 0x00);
        paddedMessage[0] = 0x00;
        paddedMessage[1] = 0x02;

        // Add random non-zero padding
        size_t paddingLength = keySize - plaintext.size() - 3;
        srand(static_cast<unsigned>(time(0)));
        for (size_t i = 0; i < paddingLength; ++i) {
            uint8_t randomByte;
            do {
                randomByte = rand() % 256;
            } while (randomByte == 0x00);  // Ensure non-zero padding
            paddedMessage[2 + i] = randomByte;
        }

        // Add the separator and plaintext
        paddedMessage[2 + paddingLength] = 0x00;
        memcpy(&paddedMessage[3 + paddingLength], plaintext.data(), plaintext.size());

        // Convert the padded message to mpz_class
        PaddedMessage = byteArrayToMpz(paddedMessage);
    }
}


void ConvertPaddedMessageToOrdinaryMessage(mpz_class &OrdinaryMessage, const mpz_class PaddedMessage) {
    // Define key size in bytes (e.g., 256 bytes for 2048-bit RSA)
    size_t keySize = 256;

    // Convert padded message to a byte array
    vector<uint8_t> paddedBytes = mpzToByteArray(PaddedMessage, keySize);

    vector<uint8_t> fullPlaintext;

    // Process the padded message in blocks
    size_t blockSize = keySize;
    size_t numBlocks = (paddedBytes.size() + blockSize - 1) / blockSize;  // Round up to get number of blocks

    for (size_t i = 0; i < numBlocks; ++i) {
        size_t blockStart = i * blockSize;
        size_t blockEnd = min(blockStart + blockSize, paddedBytes.size());
        vector<uint8_t> block(paddedBytes.begin() + blockStart, paddedBytes.begin() + blockEnd);

        // Verify padding format
        if (block[0] != 0x00 || block[1] != 0x02) {
            throw invalid_argument("Invalid padding format.");
        }

        // Find the separator (0x00) to locate the plaintext
        size_t separatorIndex = 2;
        while (separatorIndex < block.size() && block[separatorIndex] != 0x00) {
            separatorIndex++;
        }
        if (separatorIndex >= block.size()) {
            throw invalid_argument("Padding separator not found.");
        }

        // Extract the plaintext from the block (after the separator)
        vector<uint8_t> blockPlaintext(block.begin() + separatorIndex + 1, block.end());
        fullPlaintext.insert(fullPlaintext.end(), blockPlaintext.begin(), blockPlaintext.end());
    }

    // Convert the full plaintext back to an mpz_class
    OrdinaryMessage = byteArrayToMpz(fullPlaintext);
}

