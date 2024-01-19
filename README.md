# WiseRidePricer

Explore the full potential of the cutting-edge Streamlit app WiseRidePricer designed to revolutionize your car buying and selling experience! Dive into a world of comprehensive data analysis and captivating visualizations derived from a rich dataset sourced from Autoscout24. Uncover total sales figures, discern dominant brands, unravel feature correlations, and delve into temporal shifts within the automotive market. Embark on a data-driven journey with this app, where you have a plethora of regression methods at your fingertips. Predicting sales prices becomes a breeze as you select and analyze specific car characteristics. WiseRidePricer goes beyond predictions – it empowers you with model evaluation metrics, aiding in performance assessment and assisting you in choosing the optimal model for your needs. Experience the ultimate in user-friendly interfaces with a streamlined dashboard. An intuitive platform that not only effectively communicates intricate results but also forecasts prices based on your chosen models is available. Elevate your car-related decisions with this app – your go-to companion for informed choices in the dynamic world of automotive transactions!

Built with ❤️ by [jalbertofficial](https://github.com/jalbertofficial)

## What's this?

- `README.md`: This Document! To help you find your way around
- `streamlit_app.py`: The main app that gets run by [`streamlit`](https://docs.streamlit.io/)
- `requirements.txt`: Pins the version of packages needed
- `LICENSE`: Follows Streamlit's use of Apache 2.0 Open Source License
- `.gitignore`: Tells git to avoid comitting / scanning certain local-specific files
- `.streamlit/config.toml`: Customizes the behaviour of streamlit without specifying command line arguments (`streamlit config show`)
- `Makefile`: Provides useful commands for working on the project such as `run`, `lint`, `test`, and `test-e2e`
- `requirements.dev.txt`: Provides packages useful for development but not necessarily production deployment. Also includes all of `requirements.txt` via `-r`
- `pyproject.toml`: Provides a main configuration point for Python dev tools
- `.flake8`: Because `flake8` doesn't play nicely with `pyproject.toml` out of the box
- `.pre-commit-config.yaml`: Provides safeguards for what you commit and push to your repo
- `tests/`: Folder for tests to be picked up by `pytest`

## Local Setup

Assumes working python installation and some command line knowledge ([install python with conda guide](https://tech.gerardbentley.com/python/beginner/2022/01/29/install-python.html)).

```sh
# External users: download Files
git clone git@github.com:jalbertofficial/wiseridepricer.git

# Go to correct directory
cd wiseridepricer

# Run the streamlit app (will install dependencies in a virtualenvironment in the folder venv)
make run
```

Open your browser to [http://localhost:8501/](http://localhost:8501/) if it doesn't open automatically.

### Local Development

The `Makefile` and development requirements provide some handy Python tools for writing better code.
See the `Makefile` for more detail

```sh
# Run black, isort, and flake8 on your codebase
make lint
# Run pytest with coverage report on all tests not marked with `@pytest.mark.e2e`
make test
# Run pytest on tests marked e2e (NOTE: e2e tests require `make run` to be running in a separate terminal)
make test-e2e
# Run pytest on tests marked e2e and replace visual baseline images
make test-e2e-baseline
# After running tests, display the coverage html report on localhost
make coverage
```
## Deploy

For the easiest experience, deploy to [Streamlit Cloud](https://streamlit.io/cloud)

For other options, see [Streamilt deployment wiki](https://discuss.streamlit.io/t/streamlit-deployment-guide-wiki/5099)

## Credits

This package was created with Cookiecutter and the `gerardrbentley/cookiecutter-streamlit` project template.

- Cookiecutter: [https://github.com/audreyr/cookiecutter](https://github.com/audreyr/cookiecutter)
- `gerardrbentley/cookiecutter-streamlit`: [https://github.com/gerardrbentley/cookiecutter-streamlit](https://github.com/gerardrbentley/cookiecutter-streamlit)
