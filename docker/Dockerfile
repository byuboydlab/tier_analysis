FROM condaforge/miniforge3

WORKDIR /app

ADD environment.yml .
ADD tier_analysis.py .

RUN conda env create -f environment.yml
RUN conda activate sc_robustness

CMD [ "python", "./tier_analysis.py" ]