# Projeto de Recomendação de Livros com KNN

Este projeto implementa um sistema de recomendação de livros utilizando o algoritmo K-Nearest Neighbors (KNN). O sistema permite que os usuários pesquisem por livros e recebam recomendações personalizadas com base nas características dos livros.

## Tecnologias Utilizadas

- Python
- Flask (para o backend)
- Pandas (para manipulação de dados)
- Scikit-learn (para implementação do KNN)
- HTML/CSS (para a interface do usuário)

## Funcionalidades

- Pesquisa de livros pelo nome.
- Exibição de detalhes do livro, incluindo:
  - Título
  - Autor
  - Categoria
  - Preço
  - Avaliação
- Recomendações de livros com base na similaridade em características como categoria, autor, preço e avaliação.
- Exibição da distância média entre o livro selecionado e as recomendações.

## Estrutura do Projeto

/PROJETOKNN
    /templates
        index.html
        recommendations.html
    /css
        style.css
    /backend
        app.py

### Descrição dos Arquivos

- **app.py**: O arquivo principal do aplicativo Flask que gerencia as rotas e a lógica de recomendação.
- **data.csv**: Arquivo de dados que contém informações sobre os livros.
- **index.html**: Página principal onde os usuários podem pesquisar livros.
- **recommendations.html**: Página que exibe os livros recomendados.
- **style.css**: Arquivo de estilo para a interface do usuário.

## Instalação

1. **Clone o repositório e acesse a página via terminal:**
    git clone <https://github.com/JordanAquino/ProjetoKNN.git>.
    cd PROJETOKNN

3. **Instale as dependências necessárias:**
    pip install Flask pandas scikit-learn

4. **Coloque o arquivo data.csv na pasta backend.**
  
## Uso

1. **Execute o aplicativo Flask:**
    python backend/app.py

2. **Acesse o aplicativo em seu navegador:**
    http://127.0.0.1:5000

3. **Pesquise ou selecione um livro e visualize as recomendações com base na sua seleção.**


