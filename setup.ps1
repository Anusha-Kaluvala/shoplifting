# setup.ps1
# PowerShell script to set up the project environment

Write-Host "Starting project setup..." -ForegroundColor Green

# Step 1: Check if Python is installed
Write-Host "Checking for Python installation..."
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not installed. Please install Python 3.8+ and try again." -ForegroundColor Red
    exit
}

# Step 2: Create a virtual environment
Write-Host "Creating a virtual environment..."
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create a virtual environment. Please check your Python installation." -ForegroundColor Red
    exit
}

# Step 3: Activate the virtual environment
Write-Host "Activating the virtual environment..."
. .\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to activate the virtual environment. Ensure you are running this script with PowerShell." -ForegroundColor Red
    exit
}

# Step 4: Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to upgrade pip." -ForegroundColor Red
    exit
}

# Step 5: Install dependencies
Write-Host "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install dependencies. Please check the requirements file or your internet connection." -ForegroundColor Red
    exit
}

# Step 6: Notify completion
Write-Host "Setup completed successfully! Run 'python main.py' to start the application." -ForegroundColor Green
