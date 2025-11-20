# Horde Interrogate Agent

A simple backend for StableHorde's Interrogation. Aimed to be more fast and efficient, lightweight, simple and easy to set-up. Used with [Java-Horde-Bridge](https://github.com/LogicismDev/Java-Horde-Bridge)

## Features
- Image Captioning (using [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large))
- Image Interrogating (using [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))
- Image NSFW Checking (using [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32))
- Image Strip Background (using [rembg](https://github.com/danielgatis/rembg))

More to be added soon if needed.

## How to install and use

Clone the repository: `git clone https://github.com/LogicismDev/HordeInterrogateAgent`
Install Dependencies: `pip install -r requirements.txt` or `python -m pip install -r requirements.txt`
Run it: `python main.py -i {ip} -p {port}` e.g. `python main.py -i 0.0.0.0 -p 5000`

Please note that this requires the usage of Python 3.8 or higher to use the bridge.

## Flask Endpoints

### Health Endpoint - `/health`

**Method:** GET

**Example Response:** `{"status":"ok"}`
**Response Code:** 200

### Caption Endpoint - `/caption`
**Method:** POST
**Example Payload:** `{"url":"https://linktoimage.com/image.png"}`

**Example Response:** `{"caption":"example image caption"}`
**Response Code:** 200

**Example Error:** `{"error":"Example error message."}`
**Error Response Code:** 400 for invalid POST Payload. 500 for backend errors.

### Interrogation Endpoint - `/interrogate`
**Method:** POST
**Example Payload:** `{"url":"https://linktoimage.com/image.png"}`

**Example Response:** `{"interrogation":{"artists": {"Vincent Van Gogh": 0.05498}}...}`
**Response Code:** 200

**Example Error:** `{"error":"Example error message."}`
**Error Response Code:** 400 for invalid POST Payload. 500 for backend errors.

### NSFW Checking Endpoint - `/safetycheck`
**Method:** POST
**Example Payload:** `{"url":"https://linktoimage.com/image.png"}`

**Example Response:** `{"nsfw":false}`
**Response Code:** 200

**Example Error:** `{"error":"Example error message."}`
**Error Response Code:** 400 for invalid POST Payload. 500 for backend errors.

### Strip Background Endpoint - `/stripbackground`
**Method:** POST
**Example Payload:** `{"url":"https://linktoimage.com/image.png"}`

**Example Response:** `{"strip_background":"data:image/webp;base64,..."}`
**Response Code:** 200

**Example Error:** `{"error":"Example error message."}`
**Error Response Code:** 400 for invalid POST Payload. 500 for backend errors.

## Command Line Usage

| Argument Name | Argument Option | Description | Example Usage |
|--|--|--|--|
| Binding IP Address | -i | Set the flask Binding IP Address | -i 0.0.0.0 |
| Binding Port | -p | Set the flask Binding Port | -p 5000 |