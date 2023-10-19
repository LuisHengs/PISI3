# PISI3 - BSI - UFRPE

## Instalação:

<ol>
  <li>Efetue o clone do projeto: <code>CTRL+SHIFT+P > Git:Clone > Clone from GitHub > https://github.com/LuisHengs/PISI3.git</code></li>
<li>Instale o python.</li>
  
  <li>Acesse a aba "Terminal" disponível na parte inferior do VSCode.</li>

  <li>Execute a linha abaixo para criar um ambiente virtual do python para o projeto. Observe que a pasta <code>venv</code> está no <code>.gitignore</code>.<br>
    <code>python -m venv venv</code>
  </li>

  <li>Atualize o pip:<br>
    <code>python -m pip install --upgrade pip</code>
  </li>

  <li>Instale as libs necessárias para o projeto:<br>
    <code>pip install -r requirements.txt --upgrade</code>
  </li>

  <li>Rode o sistema:<br>
    <code>streamlit hello</code>
  </li>
</ol>
