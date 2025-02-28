# RAG-App
A rag app that takes a text or pdf input in a Streamlit interface, then uses OpenAI embeddings and LLM, plus ChromaDB to answer queries.

<Br> Click on the image below to watch the demo video <Br> <Br>
[![Demo video](https://img.youtube.com/vi/8_GjBYK_6tM/0.jpg)](https://youtu.be/aphlw2_UCfA) <Br> <Br>

## How to setup the project:
<ol>
  <b>
    <li>
      Prepare the OpenAI access key:
    </li>
  </b>
  
  <ul>
    <li>
      Head over to: https://platform.openai.com/
    </li>
    <li>
      Login, create an organization, and buy balance to get the key.<Br> 5$ is enough to run a lot of experiments
    </li>
  </ul>
<Br>
  <b>
    <li>
      Environment setup:
    </li>
  </b>

  <ul>
    <li>
      Download Python 3.9+ from: https://www.python.org/downloads/
    </li>
    <li>
      Create a virtual environment, by following the instructions for your operating system: https://www.geeksforgeeks.org/python-virtual-environment/
    </li>
    <li>
      Clone or download this project as zip file
    </li>
    <li>
      Open the command line/terminal, navigate to the project directory, activate the virtual environment, then run the command: <code> pip install -r requirements.txt </code>
    </li>
    
  </ul>
<Br>
  <b>
    <li>
      Run the project:
    </li>
  </b>

  <ul>
  <li>
       On the same terminal run the command: <code> chroma run </code>
    </li>
    <li>
      Open another terminal and <code> cd </code> to the same project directory and run <code>streamlit run ./app.py</code>
    </li>
    <li>
      If it does not open automatically, you can access the interface through a web browser at : <code> http://localhost:8501/ </code>
    </li>
    <li>
      Type your OpenAI access key, and click confirm
    </li>
    <li>
      You can see the video above on how to use the interface afterwards.
    </li>
    <li>
      Enjoy prompting!
    </li>
  </ul>
  
</ol>

