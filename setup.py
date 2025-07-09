"""
TradingSystem Pro パッケージセットアップファイル
"""

from setuptools import setup, find_packages
import pathlib

# パッケージディレクトリ
HERE = pathlib.Path(__file__).parent

# README.mdの内容を読み込み
README = (HERE / "README.md").read_text(encoding='utf-8')

# バージョン情報
VERSION = "1.0.0"

setup(
    name="tradingsystem-pro",
    version=VERSION,
    author="TradingSystem Pro Team",
    author_email="your-email@example.com",
    description="高度な株式取引分析・シミュレーションシステム",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/TradingSystemPro",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "mplfinance>=0.12.0",
        "pandas-ta>=0.3.0",
        "pandas-datareader>=0.10.0",
        "scipy>=1.9.0",
        "statsmodels>=0.13.0",
        "tqdm>=4.64.0",
        "python-dotenv>=0.20.0",
        "loguru>=0.6.0",
    ],
    extras_require={
        "ml": [
            "scikit-learn>=1.1.0",
            "joblib>=1.1.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.7.0",
            "plotly>=5.10.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "trading-system=src.trading_system:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["*.py"],
    },
    keywords="trading, stock analysis, technical analysis, machine learning, finance",
    project_urls={
        "Bug Reports": "https://github.com/your-username/TradingSystemPro/issues",
        "Source": "https://github.com/your-username/TradingSystemPro",
        "Documentation": "https://github.com/your-username/TradingSystemPro/wiki",
    },
)
