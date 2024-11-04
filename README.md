# LunarLanderDQN

## Building a Development Environment
### - Create the venv
* python -m venv venv
* source venv/bin/activate -> (ubuntu)
* .\venv\Scripts\activate -> (window)
* pip install --upgrade pip

### - If you use ubuntu OS
* sudo apt-get install swig
* sudo apt-get install python3-dev
* pip install -r requirements.txt

### - If you use window OS
* Install the Microsoft C++ Build Tools ([Link](https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/))
* Install the Swig ([Link](https://sourceforge.net/projects/swig/files/swigwin/swigwin-3.0.2/swigwin-3.0.2.zip/download)) -> (set the system path)
* pip install -r requirements.txt

### - Train the LunarLander.
* python main.py

### - Test the LunarLander.
* python test.py
