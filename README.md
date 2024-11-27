<html>
<head>
<style>
    body {
        font-size: 16px;
        line-height: 1.6;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    h1 {
        text-align: center;
        font-size: 28px;
        margin-bottom: 30px;
    }
    .center {
        text-align: center;
        margin: 20px 0;
    }
    p {
        text-indent: 1em;
        margin-bottom: 30px;
    }
</style>
</head>
<body>

<h1>poli-spectrum</h1>

<div class="center">
    <img src='figures/overall_figure.png' alt='Poli-Spectrum' width='600'>
</div>

<p>
Official implementation of "Poli-Spectrum: A Spectral Approach to Political Stance Detection". This repository contains codes for the training and step and other complementary codes on the english political news data, BIGNEWSBLN-Right/Left from "<i><b>POLITICS: Pretraining with Same-story Article Comparison for Ideology Prediction and Stance Detection(Liu et.al, Findings of NAACL 2022)</b></i>".
</p>

<p>
This idea was designed to overcome the limitations of the existing methods, using PCT(Political Compass Test) to detect and compare the political stances among the models, individual texts or documents, and the overall datasets. Although PCT is still a useful method to measure the political stances based on 62 closed-ended questions, my suggested idea could be much more objective and efficient due to its training data as training goal was to contain all the other political news articles which contains various political perspectives to compare objectively based on the latent space of the model. By comparing relatively subjective issues in the closed latent space, measuring could be more objective than just using off-the-shelf classifiers or sort of other tools.
</p>

<p>
The codes are implemented in Python 3.8.16 and PyTorch 1.13.1.
</p>

</body>
</html>