# Bitcoin-365

Bitcoin-365 is a comprehensive project that leverages machine learning to predict Bitcoin prices using historical data. The project utilizes both Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to explore different approaches to time series forecasting.

## Project Overview

This project scrapes historical Bitcoin price data, processes it, and then uses this data to train machine learning models for predicting future prices. The prediction models implemented are RNN and CNN, providing insights into their performance and effectiveness in financial time series prediction.

## Features

- Data scraping and preprocessing
- Visualization of Bitcoin price trends
- Predictive modeling with RNN and CNN
- Evaluation of model performance
- Environment management with Conda

## Installation

Ensure you have [Anaconda](https://www.anaconda.com/products/individual) installed to manage the project environments.

1. Clone the repository:
   ```bash
   git clone https://github.com/soyuz43/bitcoin-365.git
   cd bitcoin-365
   ```

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate bitcoin-365-env
   ```

## Usage

Run the main script to perform the data scraping, preprocessing, model training, and predictions:
```bash
python predict_bitcoin_price.py
```

### Scripts

- `predict_bitcoin_price.py`: Main script that orchestrates the loading, preprocessing, model building, training, evaluation, and plotting of predictions.
- `fetch_bitcoin_data.py`: Responsible for scraping and saving Bitcoin price data.

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](https://github.com/soyuz43/bitcoin-365/blob/main/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md]([https://github.com/soyuz43/bitcoin-365/blob/main/LICENSE](https://github.com/soyuz43/bitcoin-365/blob/main/LICENSE.md)) file for details.

## Acknowledgments

- Special thanks to the open-source community for providing the tools and libraries that make projects like this possible.
