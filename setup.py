from setuptools import setup, find_packages

setup(name="nlp_gym",
      version="0.1.0",
      description="NLPGym - A toolkit for evaluating RL agents on Natural Language Processing Tasks",
      author="Rajkumar Ramamurthy",
      author_email="raj1514@gmail.com",
      packages=find_packages(),
      python_requires='>=3.7',
      url="https://github.com/rajcscw/nlp-gym/",
      install_requires=["numpy", "torch", "nltk", "flair", "pandas", "matplotlib",
                        "seaborn", "sklearn", "tqdm", "edit_distance", "rich",
                        "gym", "pytorch-nlp", "datasets", "wget"],
      extras_require={'demo': ["tensorflow==1.15.0", "stable-baselines[mpi]"]},
      tests_requires=["pytest"])