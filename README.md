# Modelo de Reconhecimento de Ruído - Versão 1.0.1

## Descrição
Este projeto consiste em uma aplicação desenvolvida em Python com uma interface gráfica utilizando Tkinter. O modelo realiza a classificação de sons específicos, como ambulância, cachorro, caminhão de bombeiros e tráfego, a partir de arquivos de áudio no formato `.wav`. O projeto inclui funcionalidades para carregar o arquivo, gerar espectrogramas, exibir formas de onda e fazer previsões com um modelo treinado.

## Funcionalidades

### 1. Interface Gráfica (Tkinter)
- O usuário pode selecionar um arquivo de áudio para processamento.
- Exibição da previsão de classificação diretamente na interface.
- Visualização gráfica do espectrograma e da forma de onda do arquivo de áudio selecionado.

### 2. Processamento de Áudio
- **Espectrograma**: O arquivo de áudio é convertido em um espectrograma Mel que é normalizado e ajustado para uma forma adequada para o modelo.
- **Forma de Onda**: Exibe a forma de onda do áudio carregado.
  
### 3. Classificação de Áudio
- A aplicação utiliza um modelo de aprendizado de máquina treinado para classificar o áudio carregado.
- As classes possíveis incluem:
  - Ambulância
  - Cachorro
  - Caminhão de Bombeiros
  - Tráfego

### 4. Visualizações
- **Espectrograma**: A aplicação gera e exibe um espectrograma Mel do arquivo de áudio.
- **Forma de Onda**: Gera e exibe a forma de onda do áudio, facilitando a inspeção visual do som.

## Arquitetura do Projeto

### `audio_to_spectrogram(file_path, target_shape)`
- Função responsável por converter o arquivo de áudio em um espectrograma Mel, normalizá-lo e ajustá-lo ao formato adequado para a entrada no modelo.

### `create_spectrogram_image(file_path)`
- Função para criar e salvar a imagem do espectrograma a partir do áudio carregado.

### `create_waveform_image(file_path)`
- Função que gera a forma de onda do áudio e a salva em formato de imagem.

### `predict_audio(file_path)`
- Função que carrega o modelo pré-treinado, processa o áudio selecionado e realiza a previsão da classe de som.

### `open_file()`
- Função responsável por abrir a janela de diálogo para selecionar o arquivo de áudio e processá-lo para exibição na interface.

### `main()`
- Função principal que inicializa a interface Tkinter e exibe os elementos da GUI.

## Requisitos
- **Linguagem**: Python 3.x
- **Bibliotecas**:
  - `librosa`: Para processamento de áudio.
  - `tensorflow`: Para carregar e executar o modelo de reconhecimento de ruído.
  - `tkinter`: Para a interface gráfica.
  - `matplotlib`: Para geração de gráficos do espectrograma e forma de onda.
  - `Pillow`: Para manipulação de imagens e exibição na interface.

## Como Usar

1. Clone este repositório.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
```
