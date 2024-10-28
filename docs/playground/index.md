# Playground

The playground is a web app that allows you to interactively build DocETL pipelines. The playground is built with Next.js.

## Why an interactive playground?

Often, unstructured data analysis tasks are fuzzy and require iteration. You might start with a prompt, see the outputs for a sample, then realize you need to tweak the prompt or change the definition of the task you want the LLM to do. Or, you might want to create a complex pipeline that involves multiple operations, but you are unsure of what prompts you want to use for each step, so you want to build your pipeline one operation at a time.

The playground allows you to do just that.

## Installation

First, make sure you have installed the DocETL Python package from source. Fork or clone the repository, then run `make install` in the root directory:

```bash
make install
```

Then, to install the dependencies for the playground, run `make install-ui` in the root directory.

```bash
make install-ui
```

Then, run `make run-ui-dev` to start the development server.

```bash
make run-ui-dev
```

Navigate to [http://localhost:3000/playground](http://localhost:3000/playground) to access the playground. You should see the following screen:

![Playground Screenshot](../../assets/playground-screenshot.png)

