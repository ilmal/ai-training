import os
from gemini import Gemini

# os.environ["GEMINI_LANGUAGE"] = "ENG"

cookies={
    "__Secure-1PSIDCC": "AKEyXzWIIIgfylr0C_Nn1EdeD_SPVCNceOPMtfgUElrenTN8sC46w8P0U9Ifo9jvIquyyyBsTg"
}

# cookies = {"__Secure-1PSIDTS":"sidts-CjEB7F1E_MDA210ApgpjdaIc14n7GOEBS_9oo-QS5He4TEJm6K3r2R9CZBgzgpXVeBYbEAA","__Secure-3PSIDTS":"sidts-CjEB7F1E_MDA210ApgpjdaIc14n7GOEBS_9oo-QS5He4TEJm6K3r2R9CZBgzgpXVeBYbEAA","SOCS":"CAISNQgeEitib3FfaWRlbnRpdHlmcm9udGVuZHVpc2VydmVyXzIwMjQwMzI0LjA4X3AwGgJzdiACGgYIgPKnsAY","_ga":"GA1.1.551374139.1711966217","_ga_WC57KJ50ZZ":"GS1.1.1711966217.1.0.1711966218.0.0.0","NID":"512","SID":"g.a000iAi1U3VfwPtHV5eTzMYd7J6W0AdgI--yCkjpDeq7hgEZefyCfOAq6L1nxswvsE_niHbAIQACgYKAa4SAQASFQHGX2Mi098TZzjoUhdFc5jRcz58kBoVAUF8yKonRrqLRAq7tXFotdAY7ACw0076","__Secure-1PSID":"g.a000iAi1U3VfwPtHV5eTzMYd7J6W0AdgI--yCkjpDeq7hgEZefyCn66YB9T3x1rk-fhKMyawFgACgYKARQSAQASFQHGX2Mi2qFGHucNNegQWrivFZgIqRoVAUF8yKr5iRA0T9GWGgNLpYFty4K_0076","__Secure-3PSID":"g.a000iAi1U3VfwPtHV5eTzMYd7J6W0AdgI--yCkjpDeq7hgEZefyCxPWMw4l2_3Exlv90rk201gACgYKAZYSAQASFQHGX2Miwl-IgxlaPvO1NDtLOVvXlhoVAUF8yKqnh3bSij82Pqwkb3sFTymG0076","HSID":"AoB8DXRF0rd9swj8S","SSID":"AglnZ5A7SBu6tuZAi","APISID":"hMXt4nh1t--eY6gh/A2vw0aX6CezgbPHgx","SAPISID":"UPPkULJjijnw2kiA/Ac7EkUD3_CGoL3bas","__Secure-1PAPISID":"UPPkULJjijnw2kiA/Ac7EkUD3_CGoL3bas","__Secure-3PAPISID":"UPPkULJjijnw2kiA/Ac7EkUD3_CGoL3bas","SIDCC":"AKEyXzX-R8tYwOFD_H3VAgPsHax3Mfyh4A0DeKINzwu7MRwSTyf5J20_oZ5xZTP9mej_JP_VxQ","__Secure-1PSIDCC":"AKEyXzXDSIJpJv-Pg-Ge3-FQwrawyVGk6J5LMJLzMIxmkRNq4nMyqZIoDgKFdMe81iZ8XIAi","__Secure-3PSIDCC":"AKEyXzX76QehHI6SslrClqZXcv-scK1d8LrX_4btmZn-qEjkL1DbDHjuch3aGhB7I8aBOCnf"}

GeminiClient = Gemini(cookies=cookies)

# GeminiClient = Gemini(auto_cookies=True)

# GeminiClient = Gemini(auto_cookies=True, target_cookies = ["__Secure-1PSIDCC"]) 

# GeminiClient = Gemini(auto_cookies=True, target_cookies = ["__Secure-1PSID", "__Secure-1PSIDTS"]) 

# GeminiClient = Gemini(auto_cookies=True, target_cookies = ["__Secure-1PSIDCC", " __Secure-1PSID", "__Secure-1PSIDTS", "NID"]) 

response = GeminiClient.generate_content("Hello, Gemini. What's the weather like in Seoul today?")
print(response.payload)