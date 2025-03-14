FROM condaforge/miniforge3

WORKDIR /app

ADD environment.yml .
ADD tier_analysis.py .

RUN mamba env create -f environment.yml

ENTRYPOINT [ "mamba", "run", "-n", "sc_robustness", "python", "./tier_analysis.py" ]