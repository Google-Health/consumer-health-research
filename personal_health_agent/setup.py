"""Setup file for pha package installation."""

from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()



setup(
      name="pha",
      version="0.0.1",
      author="Akshay Paruchuri, Ali Heydari, Xuhai 'Orson' Xu",
      author_email="aliheydari@google.com",
      description=("Personal Health Agent: A multi-agent system for reasoning about multimodal health queries."),
      long_description=readme,
      long_description_content_type="text/markdown",
      license="CC NC-BY 4.0",
      url="https://github.com/google-health/consumer-health-research/tree/main/personal_health_agent",
      download_url="https://github.com/google-health/consumer-health-research/tree/main/personal_health_agent",
      packages=find_packages(),
      classifiers=[
                   "Development Status :: 4 - Beta",
                   "Intended Audience :: Science/Research",
                   "Programming Language :: Python :: 3.12",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence"
                   ],
      keywords=("Personal Health Agent", "Multi-Agent System", "Reasoning about Multimodal Health Queries")
      )
