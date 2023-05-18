
$$ 1.Caeser cipher $$

pt = input("Enter plain text: ")
k = int(input("Enter key: "))
ct=''
for i in pt:
    if i == " ":
      ct = ct+' '
    else:
      ct = ct + chr(((ord(i)+k)-ord('a'))%26+ord('a'))
print("Caesar cipher cipher text: ", ct)
pt2 = ''
for i in ct:
    if i == " ":
        pt2 = pt2+' '
    else:
      pt2 = pt2 + chr(((ord(i)-k)-ord('a'))%26+ord('a'))
print("Decrypted text: ", pt2)






$$ 2.Columnar $$

def val(key):
  temp = []
  for i in key:
    temp.append(i)
  temp.sort()
  keys = []
  for i in key:
    pos = temp.index(i)
    if key.count(i) > 1:
      keys.append(str(pos + 1))
      temp1 = key[0 : key.index(i)]
      temp2 = key[key.index(i) + 1 : ]
      key = temp1 + '0' + temp2
      temp.pop(pos)
      temp.insert(pos, '0')
    elif key.count(i) == 1:
      keys.append(str(pos + 1))      
  return keys

def encrypt(pt, keys):
  l = len(pt)
  l1 = len(keys)
  if (l % l1) == 0:
    rows = int(l/l1)
  else:
    temp = int(l/l1)
    for i in range(0, (temp + 1) * l1 - l):
      st = "X"
      pt += st
    rows = (len(pt))/l1

  fin = []
  i = 0
  while i < l:
    li = []
    for j in range(0, l1):
      li.append(pt[i])
      i += 1
    fin.append(li)
  print(fin)
  ct = ""
  for i in range(1, len(keys)+1):
    pos = keys.index(str(i))
    for j in fin:
      ct += j[pos]
  return ct

def decrypt(ct, keys):
  l = len(ct)
  l1 = len(keys)
  rows = int(l/l1)
  fin = []
  i = 0
  while i < l:
    li = []
    for j in range(0, l1):
      li.append('x')
      i += 1
    fin.append(li)
  i = 0
  j = 1
  while i < l:
    pos = keys.index(str(j))
    for k in range(0, rows):
      fin[k][pos] = ct[i]
      i += 1
    j += 1
    fin.append(li)
  pt = ""
  for i in range(0, rows):
    for j in range(0, l1):
      pt += fin[i][j]
  return pt

key = input("Enter key: ")
key = key.upper()
keys = val(key)
print(keys)
pt = input("Enter plain text: ")
pt = pt.replace(' ', '')
pt = pt.upper()
ct = encrypt(pt, keys)
print("Cipher text - ", ct)
pt = decrypt(ct, keys)
print("Plain text - ", pt)





$$ 3.RSA $$

import math
import string


def multiplicative_inverse(a, m):
    a=a%m; 
    for x in range(1,m) : 
        if((a*x)%m==1) : 
            return x 
    return 1
        
def generate_keypair(p, q):
    n=p*q
    print("Value of n: ",n)

    phi = (p-1)*(q-1)
    print("Value of phi(n): ", phi)
    e = 2
    while (e < phi):
        if(math.gcd(e, phi) == 1):
            break
        else:
            e = e+1
    print("value of e ",e)
    g=math.gcd(e,phi)
    while(g!=1):
        print("The number you entered is not co-prime")
        e=int(input())
        g=math.gcd(e,phi)  
    print("Value of exponent(e) entered is: ", e)
    d = multiplicative_inverse(e, phi)
    return (e,n),(d,n)


def encrypt(public_key, to_encrypt):
    key, n = public_key

    cipher=pow(to_encrypt,key)%n
    return cipher


def decrypt(private_key, to_decrypt):
    key, n = private_key

    decrypted=pow(to_decrypt,key)%n
    return decrypted

p=int(input("Enter prime p: "))
q=int(input("Enter prime q (!=p): "))

while(p==q):
    p=int(input("Enter prime p: "))
    q=int(input("Enter prime q (!=p): "))
    
print("Prime number p: ",p)
print("Prime number q: ",q)

print("Generating Public/Private key-pairs!")
public, private = generate_keypair(p, q)
print("Your public key is (e,n) ", public)
print("Your private key is (d,n) ", private)

message = int(input("Enter the message: "))

print("Encrypted message (Cipher Text): ",encrypt(public,message))

message_c = int(input("Enter cipher message: "))
print("Decrypted message (Plain Text): ", decrypt(private,message_c))







$$ 4.Diffie-Hellman $$

import random as r

p = int(input("Enter value of p: "))
g = int(input("Enter value of g: "))
A = r.randint(3, 1000)
B = r.randint(3, 1000)

print("A - ", A)
print("B - ", B)

X_a = (g**A) % p
X_b = (g**B) % p

A_k = (X_b**A) % p
B_k = (X_a**B) % p

if A_k == B_k:
  print("Symmetric Key = ", A_k)







$$ 5.Playfair technique $$


def create_matrix(key):
    key = key.upper()
    matrix = [[0 for i in range (5)] for j in range(5)]
    letters_added = []
    row = 0
    col = 0

    for letter in key:
        if letter not in letters_added:
            matrix[row][col] = letter
            letters_added.append(letter)
        else:
            continue
        if (col==4):
            col = 0
            row += 1
        else:
            col += 1

    for letter in range(65,91):
        if letter==74:
                continue
        if chr(letter) not in letters_added:
            letters_added.append(chr(letter))
            
    index = 0
    for i in range(5):
        for j in range(5):
            matrix[i][j] = letters_added[index]
            index+=1
    return matrix

def separate_same_letters(message):
    index = 0
    while (index<len(message)):
        l1 = message[index]
        if index == len(message)-1:
            message = message + 'X'
            index += 2
            continue
        l2 = message[index+1]
        if l1==l2:
            message = message[:index+1] + "X" + message[index+1:]
        index +=2   
    return message

def indexOf(letter,matrix):
    for i in range (5):
        try:
            index = matrix[i].index(letter)
            return (i,index)
        except:
            continue

def encrypt(key, message):
    inc = 1
    matrix = create_matrix(key)
    message = message.upper()
    message = message.replace(' ','')    
    message = separate_same_letters(message)
    cipher_text=''
    for (l1, l2) in zip(message[0::2], message[1::2]):
        row1,col1 = indexOf(l1,matrix)
        row2,col2 = indexOf(l2,matrix)
        if row1==row2:
            cipher_text += matrix[row1][(col1+inc)%5] + matrix[row2][(col2+inc)%5]
        elif col1==col2:
            cipher_text += matrix[(row1+inc)%5][col1] + matrix[(row2+inc)%5][col2]
        else:
            cipher_text += matrix[row1][col2] + matrix[row2][col1]
    
    return cipher_text

def decrypt(key, message):
    inc = 1
    matrix = create_matrix(key)
    message = message.upper() 
    plain_text=''
    for (l1, l2) in zip(message[0::2], message[1::2]):
        row1,col1 = indexOf(l1,matrix)
        row2,col2 = indexOf(l2,matrix)
        if row1==row2:
            plain_text += matrix[row1][(col1-inc)%5] + matrix[row2][(col2-inc)%5]
        elif col1==col2:
            plain_text += matrix[(row1-inc)%5][col1] + matrix[(row2-inc)%5][col2]
        else:
            plain_text += matrix[row1][col2] + matrix[row2][col1]
    
    return plain_text

key = input("Enter key - ")
print("Matrix")
print(create_matrix(key))
print ('\nEncrypting')
pt = input("Enter plain text - ")
print(encrypt(key, pt))
print ('\nDecrypting')
ct = input("Enter cipher text - ")
print(decrypt(key, ct))








$$ 6.Merkle Root $$


from hashlib import sha256

def hash(x):
  ans = sha256(x.encode("utf-8")).hexdigest()
  return ans

def hash_value(h):
  h1 = []
  if len(h) % 2 == 0:
    for i in range(0, len(h), 2):
      text = h[i] + h[i + 1]
      h1.append(hash(text))
  else:
    for i in range(0, len(h) - 1, 2):
      text = h[i] + h[i + 1]
      h1.append(hash(text))
    h1.append(h[len(h) - 1])

  return h1

para = input("Enter para (use '.' to seperate lines): ")

l = para.split('.')
count = len(l)

if count % 8 != 0 :
  temp = int(count / 8)
  for i in range(0, (temp + 1) * 8 - count):
    l.append(l[count - 1])

h = list(map(hash, l))
length = len(h)

while length > 1:
  h = hash_value(h)
  length = len(h)

print("\n\nMerkle root - ", h[0])








$$ 7.Electronic Code Block $$


def lhs(l):
  temp=[]
  t=2
  for i in range(len(l)-t):
    temp.append(l[i+t])
  temp.extend(l[0:t])
  print(temp)
  ct.extend(temp)

def rhs(l):
  temp=[]
  t=2
  for i in range(len(l)):
    temp.append(l[i-t])
  pt.extend(temp)

para="Hello my name is dash. I am a third year student in computer engineering in DJ Sanghvi college. My graduation year is 2024."
para=para.upper()

#Encryption
ascii=""
binary=""
for i in para:
  ascii=ascii+str(ord(i))
ascii=int(ascii)
binary=bin(ascii)
binary=binary[2:] #removing 0b
print("Ascii value: ",ascii)
print("Binary value: ",binary)
print("Encrpted paragraph is: ")
pos=0
i=0
block=[]
ct=[]
pt=[]

while binary and pos<len(binary):
  while i<128 and pos<len(binary):
    block.append(binary[pos])
    pos=pos+1
    if pos%128==0:
      lhs(block)
      block=[]
      i=0
      continue
lhs(block)

#Decryption
block=[]
pos=0
i=0
while ct and pos<len(ct):
  while i<128 and pos<len(ct):
    block.append(ct[pos])
    pos=pos+1
    if pos%128==0:
      rhs(block)
      block=[]
      i=0
      continue
rhs(block)
plaintext=""
for i in pt:
  plaintext=plaintext+i
ascii=int(plaintext,2)
ascii=str(ascii)
i=0
ans=""
while i<len(ascii):
  ans=ans+chr(int(ascii[i]+ascii[i+1]))
  i=i+2
print("\nDecrypted Text is:",ans)








$$ 8. Hill Cipher $$


import numpy as np

def hill_encrypt(plain_text, key):
    plain_text = plain_text.upper().replace(" ", "")

    if len(plain_text) % key.shape[0] != 0:
        plain_text += "X" * (key.shape[0] - len(plain_text) % key.shape[0])

    plain_text = np.array(list(plain_text)).reshape((-1, key.shape[0]))

    cipher_text = ""
    for block in plain_text:
        block_num = np.array([ord(ch) - 65 for ch in block])

        cipher_num = np.dot(key, block_num) % 26

        cipher_text += "".join([chr(int(num + 65)) for num in cipher_num])

    return cipher_text


def hill_decrypt(ciphertext, key):
    key_inverse = np.linalg.inv(key)
    
    key_det = np.round(np.linalg.det(key)).astype(int)
    
    det_inverse = 0
    for i in range(26):
        if (i * key_det) % 26 == 1:
            det_inverse = i
            break
    
    if det_inverse == 0:
        return None
    
    key_mod_inverse = np.mod(det_inverse * key_det * key_inverse, 26)
    
    ciphertext = ''.join(filter(str.isalpha, ciphertext.upper()))
    
    while len(ciphertext) % key.shape[0] != 0:
        ciphertext += 'X'
    
    blocks = [ciphertext[i:i+key.shape[0]] for i in range(0, len(ciphertext), key.shape[0])]
    
    plaintext = ''
    for block in blocks:
        block_indices = [ord(char) - ord('A') for char in block]
        block_matrix = np.array(block_indices).reshape(key.shape[0], -1)
        block_plaintext_matrix = np.mod(key_mod_inverse @ block_matrix, 26)
        block_plaintext_indices = block_plaintext_matrix.flatten().tolist()
        block_plaintext = ''.join([chr(int(index) + ord('A')) for index in block_plaintext_indices])
        plaintext += block_plaintext
    
    return plaintext
    

n = int(input("Enter order of key : "))

key = []
for i in range(n):
    row = []
    for j in range(n):
        ans = int(input("Enter value : "))
        row.append(ans)
    key.append(row)
    
key = np.array(key)
print(key)

while(1):
    choice=int(input("\n 1.Encryption \n 2.Decryption \n 3.EXIT \n Choice : "))
    if choice==1:
        plain_text = input("Enter message : ")
        cipher_text = hill_encrypt(plain_text, key)
        print(cipher_text)
    elif choice==2:
        cipher_text = input("Enter cipher text : ")
        decrypted_text = hill_decrypt(cipher_text, key)
        print(decrypted_text)
    elif choice==3:
        exit()
    else:
        print("Choose correct choice")








$$ 9. Own Algo $$



def encryption(text):
    
    text = list(text)
    for i in range(1, len(text), 2):
        text[i] = text[len(text) - i - 1]
    

    key = int(input("Please enter key:"))
    if len(text) % key != 0:
        padding = key - (len(text) % key)
        text += ['#'] * padding
    

    groups = [text[i:i+key] for i in range(0, len(text), key)]
    
 
    transposed = []
    for i in range(key):
        transposed.append([group[i] for group in groups])
    

    encrypted = ''.join(''.join(group) for group in transposed)
    print("Encrypted text is:", encrypted)
    
    
    decryption(encrypted, key)


def decryption(text, key):
    # Step 1: Split the text into groups
    groups = [list(text[i:i+key]) for i in range(0, len(text), key)]
    
    # Step 2: Transpose the groups
    transposed = []
    for i in range(len(groups[0])):
        transposed.append([group[i] for group in groups])
    
    # Step 3: Reverse characters at odd indices
    for i in range(len(transposed)):
        if i % 2 != 0:
            transposed[i] = transposed[i][::-1]
    
    # Step 4: Flatten the transposed groups to get the decrypted text
    decrypted = ''.join(''.join(group) for group in transposed)
    decrypted = decrypted.replace('#', '')  # Remove padding characters
    
    print("Decrypted text is:", decrypted)


encryption("aaravsharma")









$$ 10.Digital Signature $$


import random
from hashlib import sha256


def coprime(a, b):
    while b != 0:
        a, b = b, a % b
    return a
    
    
def extended_gcd(aa, bb):
    lastremainder, remainder = abs(aa), abs(bb)
    x, lastx, y, lasty = 0, 1, 1, 0
    while remainder:
        lastremainder, (quotient, remainder) = remainder, divmod(lastremainder, remainder)
        x, lastx = lastx - quotient*x, x
        y, lasty = lasty - quotient*y, y
    return lastremainder, lastx * (-1 if aa < 0 else 1), lasty * (-1 if bb < 0 else 1)

#Euclid's extended algorithm for finding the multiplicative inverse of two numbers    
def modinv(a, m):
	g, x, y = extended_gcd(a, m)
	if g != 1:
		raise Exception('Modular inverse does not exist')
	return x % m    

        
def is_prime(num):
    if num == 2:
        return True
    if num < 2 or num % 2 == 0:
        return False
    for n in range(3, int(num**0.5)+2, 2):
        if num % n == 0:
            return False
    return True


def generate_keypair(p, q):
    if not (is_prime(p) and is_prime(q)):
        raise ValueError('Both numbers must be prime.')
    elif p == q:
        raise ValueError('p and q cannot be equal')

    n = p * q

    #Phi is the totient of n
    phi = (p-1) * (q-1)

    #Choose an integer e such that e and phi(n) are coprime
    e = random.randrange(1, phi)

    #Use Euclid's Algorithm to verify that e and phi(n) are comprime 
    g = coprime(e, phi)
  
    while g != 1:
        e = random.randrange(1, phi)
        g = coprime(e, phi)

    #Use Extended Euclid's Algorithm to generate the private key
    d = modinv(e, phi)

    #Return public and private keypair
    #Public key is (e, n) and private key is (d, n)
    return ((e, n), (d, n))


def encrypt(privatek, plaintext):
    #Unpack the key into it's components
    key, n = privatek

    #Convert each letter in the plaintext to numbers based on the character using a^b mod m
            
    numberRepr = [ord(char) for char in plaintext]
    print("Number representation before encryption: ", numberRepr)
    cipher = [pow(ord(char),key,n) for char in plaintext]
    
    #Return the array of bytes
    return cipher


def decrypt(publick, ciphertext):
    #Unpack the key into its components
    key, n = publick
       
    #Generate the plaintext based on the ciphertext and key using a^b mod m
    numberRepr = [pow(char, key, n) for char in ciphertext]
    plain = [chr(pow(char, key, n)) for char in ciphertext]

    print("Decrypted number representation is: ", numberRepr)
    
    #Return the array of bytes as a string
    return ''.join(plain)
    
    
def hashFunction(message):
    hashed = sha256(message.encode("UTF-8")).hexdigest()
    return hashed
    
    
def verify(receivedHashed, message):
    ourHashed = hashFunction(message)
    if receivedHashed == ourHashed:
        print("Verification successful: ", )
        print(receivedHashed, " = ", ourHashed)
    else:
        
        print("Verification failed")
        print(receivedHashed, " != ", ourHashed)
        

def main():
    p = int(input("Enter a prime number (17, 19, 23, etc): "))
    q = int(input("Enter another prime number (Not one you entered above): "))   
    #p = 17
    #q=23
    
    
    print("Generating your public/private keypairs now . . .")
    public, private = generate_keypair(p, q)
    
    print("Your public key is ", public ," and your private key is ", private)
    message = input("Enter a message to encrypt with your private key: ")
    print("")

    hashed = hashFunction(message)
    
    print("Encrypting message with private key ", private ," . . .")
    encrypted_msg = encrypt(private, hashed)   
    print("Your encrypted hashed message is: ")
    print(''.join(map(lambda x: str(x), encrypted_msg)))
    #print(encrypted_msg)
    
    print("")
    print("Decrypting message with public key ", public ," . . .")

    decrypted_msg = decrypt(public, encrypted_msg)
    print("Your decrypted message is:")  
    print(decrypted_msg)
    
    print("")
    print("Verification process . . .")
    verify(decrypted_msg, message)
   
main()    






$$ 11.Buffer Overflow $$


#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	char buffer[5];
	if (argc < 2)
	{
			printf("strcpy() NOT executed....\n");
			printf("Syntax: %s <characters>\n", argv[0]);
			exit(0);
	}
	strcpy(buffer, argv[1]);
	printf("buffer content= %s\n", buffer);
	printf("strcpy() executed...\n");

	return 0;
}
