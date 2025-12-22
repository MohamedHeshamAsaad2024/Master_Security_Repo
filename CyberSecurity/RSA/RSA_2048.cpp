/*************************************************************
 * File Name: RSA_2048
 * Description:
 *  Implementation of RSA 2048 with padding scheme PKCS v1.5
 *  The implementation does the following:
 *  - Generate keys and test for primality using GCD and Jacobi
 *  - Obtains the multiplicative inverse using extended euclidean
 *  - Perform PCKS v1.5 Padding
 *  - Performs encryption and decryption using Indian Algorithm
 *  
 * Implemented by 
 *  Abdelrahman Gamil
 *  Eslam abdelsamea
 *  Mohamed eldehamy 
 *  Mohamed Hesham Kamaly
 ************************************************************/
/*************************************************************
 *                      I N C L U D E S                      *
 *************************************************************/
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <cassert>
#include <tuple>
#include <gmpxx.h>

/*************************************************************
 *                     N A M E S P A C E S                   *
 *************************************************************/
using namespace std;

/*************************************************************
 *                     P R O T O T Y P E S                   *
 *************************************************************/
void generateKeys(mpz_class &Public_Modulus, mpz_class &Public_Exponent, mpz_class &Private_Exponent);
mpz_class computePhi(const mpz_class PrimeNumber_1, const mpz_class PrimeNumber_2) ;
void computeGCD(mpz_class result, const mpz_class num1, const mpz_class num2);
void generateTwoPrimeNumbers(mpz_class &PrimeNumber_1, mpz_class &PrimeNumber_2);
void ConvertOrdinaryMessageToPaddedMessage(mpz_class &PaddedMessage, const mpz_class OrdinaryMessage);
void ConvertPaddedMessageToOrdinaryMessage(mpz_class &OrdinaryMessage, const mpz_class PaddedMessage);
void generateRandomNumber(mpz_class &randomNumber);
mpz_class jacobi(mpz_class r, mpz_class p);
bool isPrime(mpz_class p, mpz_class k);
mpz_class RSA_PKCS_1_encrypt(const mpz_class &message, const mpz_class &publicExponent, const mpz_class &modulus);
mpz_class RSA_PKCS_1_decrypt(const mpz_class &cipher, const mpz_class &privateExponent, const mpz_class &modulus);
mpz_class modularExponentiation(mpz_class base, mpz_class exponent, mpz_class modulus);
mpz_class modularInverse(mpz_class PublicExponent, mpz_class Phi);
mpz_class extendedEuclidean(mpz_class FirstInteger, mpz_class SecondInteger, mpz_class &Coefficient_FirstInteger, mpz_class &Coefficient_SecondInteger);

/*************************************************************
 *                           M A I N                         *
 *************************************************************/
int main() 
{
    /* Declare placeholder to take string input from user interactively */
    std::string inputMessage;

    /* Declare placeholder for the input plain text message */
    mpz_class message;

    /* Declare placeholder for the padded plain text message */
    mpz_class Padded_message;

    /* Declare placeholder for the cipher text message after encryption of padded plain text */
    mpz_class ciphertext;
    
    /* Declare placeholder for the decrypted padded plain text message */
    mpz_class decryptedPaddedMessage;

    /* Declare placeholder for the decrypted plain text message */
    mpz_class decryptedMessage;

    /* Declare placeholders for the keys */
    mpz_class Public_Modulus;
    mpz_class Public_Exponent;
    mpz_class Private_Exponent;

    // Display a message indicating that RSA key generation is in progress
    cout << "Generating RSA keys (2048-bit)..." << endl;

    // Generate RSA keys (public modulus, public exponent, and private exponent)
    generateKeys(Public_Modulus, Public_Exponent, Private_Exponent);

    // Display the generated RSA keys
    gmp_printf("\nPublic Key (Public_Modulus, Public_Exponent):\n(%Zx, %Zd)\n", Public_Modulus, Public_Exponent);
    gmp_printf("\nPrivate Key (Private_Exponent):\n(%Zx)\n", Private_Exponent);

    std::cout << "\nEnter a number in hex:\n";
    std::cin >> inputMessage; // Read input as a string

    // Set the message to be encrypted. 
    // The second argument is set to 10 in case of decimal value and set to 16 in case of hex value
    message.set_str(inputMessage, 16);

    // Display the original message
    gmp_printf("\nOriginal message in hexadecimal:\n%Zx\n", message.get_mpz_t());

    // Convert the ordinary message to a padded message
    ConvertOrdinaryMessageToPaddedMessage(Padded_message, message);
    gmp_printf("\nPadded message in hexadecimal:\n%Zx\n", Padded_message.get_mpz_t());

    // Encrypt the padded message using the public key
    ciphertext = RSA_PKCS_1_encrypt(Padded_message, Public_Exponent, Public_Modulus);
    gmp_printf("\nEncrypted message:\n%Zx\n", ciphertext.get_mpz_t());

    // Decrypt the encrypted message using the private key
    decryptedPaddedMessage = RSA_PKCS_1_decrypt(ciphertext, Private_Exponent, Public_Modulus);
    gmp_printf("\nDecrypted padded message:\n%Zx\n", decryptedPaddedMessage.get_mpz_t());

    // Convert the decrypted padded message back to the original message
    ConvertPaddedMessageToOrdinaryMessage(decryptedMessage, decryptedPaddedMessage);
    gmp_printf("\nDecrypted message:\n%Zx\n", decryptedMessage.get_mpz_t());

    return 0;
}

/*************************************************************
 *                       F U N C T I O N S                   *
 *************************************************************/
// Generates RSA keys: public and private (Stub function for demonstration purposes)
// Kamaly's responsibility
void generateKeys(mpz_class &Public_Modulus, mpz_class &Public_Exponent, mpz_class &Private_Exponent) 
{
    // Declare placeholders to hold the randomly generated two prime numbers
    mpz_class Generted_PrimeNumber_1, Generted_PrimeNumber_2, phi;

    // Generate two prime numbers
    generateTwoPrimeNumbers(Generted_PrimeNumber_1, Generted_PrimeNumber_2);

    // Calculate Public_Modulus
    Public_Modulus = Generted_PrimeNumber_1 * Generted_PrimeNumber_2;

    // Set  Public_Modulus to make calculation faster and optmize implementation
    Public_Exponent = "65537";

    // Function to calculate Euler's Totient function for n = p * q
    phi = computePhi(Generted_PrimeNumber_1, Generted_PrimeNumber_2);

    // Calculate Private_Exponent
    Private_Exponent = modularInverse(Public_Exponent, phi);

   return;
}

// Function to calculate Euler's Totient function for n = p * q
mpz_class computePhi(const mpz_class PrimeNumber_1, const mpz_class PrimeNumber_2) 
{
    mpz_class PrimeNumber_1_minus_1, PrimeNumber_2_minus_1, result;

    // Compute PrimeNumber_1 - 1 and PrimeNumber_2 - 1
    PrimeNumber_1_minus_1 = PrimeNumber_1 - 1;
    PrimeNumber_2_minus_1 = PrimeNumber_2 - 1;

    // Compute phi(n) = (PrimeNumber_1 - 1) * (PrimeNumber_2 - 1)
    result = PrimeNumber_1_minus_1 * PrimeNumber_2_minus_1;

    return result;
}

//Function to generate two different prime numbers
void generateTwoPrimeNumbers(mpz_class &PrimeNumber_1, mpz_class &PrimeNumber_2)
{
    mpz_class randomNumber;  // mpz_class variable to store generated random numbers
    
    // Generate a prime number for PrimeNumber_1
    do {
        generateRandomNumber(randomNumber);  // Generate a random number
    } while (!isPrime(randomNumber, 5));  // Keep generating until it's prime
    PrimeNumber_1 = randomNumber;  // Assign the prime number to PrimeNumber_1

    // Generate a prime number for PrimeNumber_2
    do {
        generateRandomNumber(randomNumber);  // Generate a random number
    } while (!isPrime(randomNumber, 5));  // Keep generating until it's prime
    PrimeNumber_2 = randomNumber;  // Assign the prime number to PrimeNumber_2
}


// Function to generate a random 256-byte number
void generateRandomNumber(mpz_class &randomNumber)  
{
    // Initialize GMP random state
    gmp_randstate_t state;
    gmp_randinit_default(state);

    // Seed the random state with current time
    std::random_device rd;
    unsigned long seed = rd();
    gmp_randseed_ui(state, seed);

    // Generate a random number of 1024 such that when multiplicated by another 1024 bit it produces 2048 exponent
    mpz_urandomb(randomNumber.get_mpz_t(), state, 1024);

    // Clear random state
    gmp_randclear(state);
}

// Function to calculate the Jacobi symbol (r/p)
mpz_class jacobi(mpz_class r, mpz_class p) 
{
    if (r == 0) return (p == 1) ? 1 : 0;  // Base case: Jacobi symbol is 0 if r is 0, unless p is 1
    if (r == 1) return 1;  // Base case: Jacobi symbol is 1 if r is 1

    mpz_class result = 1;
    if (r % 2 == 0) {  // Check if r is even
        result = jacobi(r / 2, p);  // Recursive call
        mpz_class Even_power = (p * p - 1) / 8;  // Calculate exponent
        if (Even_power.get_ui() % 2 == 1) {  // Adjust sign if exponent is odd
            result = -result;
        }
    } else {  // When r is odd
        result = jacobi(p % r, r);  // Recursive call
        mpz_class Odd_power = (r - 1) * (p - 1) / 4;  // Calculate exponent
        if (Odd_power.get_ui() % 2 == 1) {  // Adjust sign if exponent is odd
            result = -result;
        }
    }
    return result;
}


// Function to compute GCD (Greatest Common Divisor) of a and b
mpz_class computeGCD(mpz_class a, mpz_class b) 
{
    while (b != 0) 
    {  // Loop until b becomes 0
        mpz_class t = b;  // Store b in temporary variable
        b = a % b;  // Update b to a modulo b
        a = t;  // Update a to temporary variable
    }
    return a;  // Return the GCD
}

// Function to perform the Solovay-Strassen primality test
bool isPrime(mpz_class p, mpz_class k = 5) 
{
    if (p == 2) return true;  // 2 is prime
    if (p < 2 || p % 2 == 0) return false;  // No even number less than 2 is prime

    for (mpz_class i = 0; i < k; ++i) 
    {
        mpz_class r = rand() % (p - 2) + 2;  // Random integer in range [2, p-1]
        
        // Check if r and p are coprime
        if (computeGCD(r, p) != 1)  return false;
        
        mpz_class jacobian = (p + jacobi(r, p)) % p;  // Calculate Jacobi symbol and adjust to be positive
        mpz_class mod = modularExponentiation(r, (p - 1) / 2, p);  // Compute r^((p-1)/2) mod p

        if (jacobian == 0 || mod != jacobian)  // If condition fails, p is composite
            return false;
    }
    return true;  // Probably prime if all tests passed
}

/**
 * Function to compute the GCD and coefficients (x, y) such that:
 * ax + by = gcd(a, b)
 * @param FirstInteger              - First integer
 * @param SecondInteger             - Second integer
 * @param Coefficient_FirstInteger  - Reference to coefficient First integer
 * @param Coefficient_SecondInteger - Reference to coefficient SecondInteger
 * @return gcd(FirstInteger, SecondInteger)
 */
mpz_class extendedEuclidean(mpz_class FirstInteger, mpz_class SecondInteger, mpz_class &Coefficient_FirstInteger, mpz_class &Coefficient_SecondInteger) 
{
    // Base case: if b is 0, gcd is a, and x = 1, y = 0
    if (SecondInteger == 0) {
        Coefficient_FirstInteger  = 1;
        Coefficient_SecondInteger = 0;
        return FirstInteger;
    }

    // Recursive call
    mpz_class x1, y1; // Temporary coefficients
    mpz_class gcd = extendedEuclidean(SecondInteger, FirstInteger % SecondInteger, x1, y1);

    // Update x and y using the results of the recursive call
    Coefficient_FirstInteger = y1;
    Coefficient_SecondInteger = x1 - (FirstInteger / SecondInteger) * y1;

    return gcd;
}

/**
 * Function to compute the modular inverse of a modulo m using the Extended Euclidean Algorithm.
 * The modular inverse exists if and only if gcd(a, m) = 1.
 * @param PublicExponent - The number to find the inverse of
 * @param Phi - The modulus
 * @return The modular inverse of a modulo m, or -1 if no inverse exists
 */
mpz_class modularInverse(mpz_class PublicExponent, mpz_class Phi) 
{
    mpz_class x, y;
    mpz_class gcd = extendedEuclidean(PublicExponent, Phi, x, y);

    // If gcd(a, m) != 1, modular inverse does not exist
    if (gcd != 1) {
        cout << "Modular inverse does not exist for " << PublicExponent << " modulo " << Phi << endl;
        return -1;
    }

    // Ensure the result is positive
    return (x % Phi + Phi) % Phi;
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

/*************************************************************
 * Function Name: RSA_PKCS_1_encrypt
 * Descriotion:
 *  This function acts as a wrapper for the modular exponentiation
 *  to be used for encryption
 * Arguments:
 *  message (mpz_class): The message to be encrypted
 *  publicExponent (mpz_class): The public exponent
 *  modulus (mpz_class): The modulus
 * Return:
 *  cipherText (mpz_class): The encrypted message
 ************************************************************/
mpz_class RSA_PKCS_1_encrypt(const mpz_class& message, const mpz_class& publicExponent, const mpz_class& modulus) 
{
    mpz_class cipherText = modularExponentiation(message, publicExponent, modulus);

    return cipherText;
}

/*************************************************************
 * Function Name: RSA_PKCS_1_decrypt
 * Descriotion:
 *  This function acts as a wrapper for the modular exponentiation
 *  to be used for decryption
 * Arguments:
 *  cipher (mpz_class): The message to be decrypted
 *  privateExponent (mpz_class): The private exponent
 *  modulus (mpz_class): The modulus
 * Return:
 *  decryptedText (mpz_class): The decrypted message
 ************************************************************/
mpz_class RSA_PKCS_1_decrypt(const mpz_class& cipher, const mpz_class& privateExponent, const mpz_class& modulus)
{
    mpz_class decryptedText = modularExponentiation(cipher, privateExponent, modulus);

    return decryptedText;
}

/*************************************************************
 * Function Name: modularExponentiation
 * Descriotion:
 *  This function performs modular exponentiation using the
 *  Indian algorithm. Given (base ^ exponent) mode (modulus),
 *  the algorithm works as follows:
 *      1- Initialization Step: Start with Result = 1
 *      2- Start from the LSB of the exponent
 *      3- While the exponent is not zero
 *          3.1- If the current bit is 1, update result by
 *               multiplying the result by base
 *          3.2- Reduce the result to the modulus value
 *          3.3- Update the base by multiplying it by itself
 *          3.4- Shift the exponent left (Halving it)
 *      4- The final result is the data of Result
 * Arguments:
 *  base (mpz_class): TThe base of the exponentiation
 *  exponent (mpz_class): The exponent value
 *  modulus (mpz_class): The modulus for reduction
 * Return:
 *  result (mpz_class): The final result of the modular
 *                       exponentiation
 ************************************************************/
mpz_class modularExponentiation(mpz_class base, mpz_class exponent, mpz_class modulus)
{
    /* Initialization Step: Start with result of 1 */
    mpz_class result = 1;

    /* Initialization Step: Start with base  of base % modulus */
    base = base % modulus;

    /* Begin as long as the exponent is not equal to zero */
    while (exponent > 0)
    {
        /* Check if the current bit (LSB) is set to 1*/
        if (exponent % 2 == 1)
            /* Update the result by multipling the previous result by the base and reduce it by the modulus */
            result = (result * base) % modulus;
        
        /* Update the base by multipling the base by itself and reduce it by the modulus */
        base = (base * base) % modulus;

        /* Shift right to consider the next bit. Essentially, halving the square */
        exponent = exponent >> 1;
    }

    /* Return the result */
    return result;
}