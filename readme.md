To set up and run the AI chat interface with real-time data on your personal computer, follow these steps:

1. **Install Ollama:**

   - **For macOS:**
     - Visit the [Ollama download page](https://ollama.com/download) and download the installer for macOS.
     - Open the downloaded `.dmg` file and follow the on-screen instructions to install Ollama.
   - **For Windows:**
     - Go to the [Ollama download page](https://ollama.com/download/windows) and download the installer for Windows.
     - Run the downloaded `.exe` file and follow the installation prompts.
   - **For Linux:**
     - Open a terminal window.
     - Execute the following command to download and install Ollama:
       ```bash
       curl -fsSL https://ollama.com/install.sh | sh
       ```
     - Follow any additional instructions provided during the installation process.

2. **Download the Dolphin Model:**

   - After installing Ollama, download the Dolphin model using the command line:
     - Open a terminal or command prompt.
     - Run the following command to download the Dolphin model:
       ```bash
       ollama pull dolphin-llama3
       ```
     - Wait for the download to complete.

3. **Install Visual Studio Code (VS Code):**

   - Visit the [Visual Studio Code website](https://code.visualstudio.com/).
   - Download the installer appropriate for your operating system (Windows, macOS, or Linux).
   - Run the installer and follow the on-screen instructions to complete the installation.

4. **Set Up a Python Virtual Environment:**

   - **Ensure Python is Installed:**
     - Open a terminal or command prompt.
     - Check if Python is installed by running:
       ```bash
       python --version
       ```
     - If Python is not installed, download and install it from the [official Python website](https://www.python.org/).
   - **Create a Virtual Environment:**
     - Navigate to your project directory:
       ```bash
       cd /path/to/your/project
       ```
     - Create a virtual environment named `venv`:
       ```bash
       python -m venv venv
       ```
   - **Activate the Virtual Environment:**
     - **On Windows:**
       ```bash
       venv\Scripts\activate
       ```
     - **On macOS and Linux:**
       ```bash
       source venv/bin/activate
       ```
     - After activation, your command prompt will show `(venv)` indicating that the virtual environment is active.

5. **Install Required Python Packages:**

   - With the virtual environment activated, install the necessary packages:
     ```bash
     pip install streamlit requests beautifulsoup4 pydantic langchain
     ```
   - These packages are essential for the AI chat interface to function correctly.

6. **Run the Streamlit Application:**
   - Ensure your virtual environment is activated.
   - Navigate to the directory containing your Streamlit application script (e.g., `app.py`).
   - Run the application using Streamlit:
     ```bash
     streamlit run app.py
     ```
   - A new browser window or tab should open, displaying your AI chat interface.

Not important (ignore this part)

```bash

python -m venv .venv

.\.venv\Scripts\Activate

streamlit run app.py
```
