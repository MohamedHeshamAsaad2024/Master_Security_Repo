#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <cassert>
#include <tuple>
#include <cstdint>
#include <gmpxx.h>

long long RSA_ModularExponentiation(long long base, long long exponent, long long modulus)
{
    /* Initialization Step: Start with result of 1 */
    long long result = 1;

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

mpz_class RSA_ModularExponentiation(mpz_class base, mpz_class exponent, mpz_class modulus)
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

int main()
{
    // Small Numbers
    long long base = 31;
    long long publicExponent = 17;
    long long modulus = 2773;
    long long privateExponent = 157;

    // Small Encrypt
    long long encryptedOutput = RSA_ModularExponentiation(base, publicExponent, modulus);
    std::cout << encryptedOutput << std::endl;

    // Small Decrypt
    long long decryptedOutput = RSA_ModularExponentiation(encryptedOutput, privateExponent, modulus); 
    std::cout << decryptedOutput << std::endl;

    // Large Numbers
    mpz_class baseLarge("31");
    mpz_class exponentLarge("17");
    mpz_class modulusLarge("2773");
    mpz_class privateExponentLarge("157");

    // Large Encrypt
    mpz_class encryptedOutputLarge = RSA_ModularExponentiation(baseLarge, exponentLarge, modulusLarge);
    std::cout << encryptedOutputLarge << std::endl;

    // Large Decrypt
    mpz_class decryptedOutputLarge = RSA_ModularExponentiation(encryptedOutputLarge, privateExponentLarge, modulusLarge);
    std::cout << decryptedOutputLarge << std::endl;

    return 0;
}
