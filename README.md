*This project has been created as part of the 42 curriculum by joshde-s*

# Call Me Maybe

## Description

### Overview

The primary goal of this project is to use llm model(s) to generate tokens
    that are based on function calls. The important part is that the output
    tokens from the LLM are constrained by the program to create correct
    function calls that could be theoretically used on the given prompts.

### Project Goals and Specifications

#### Step 1 - Parser and Input Validation


#### Step 2 - Vocab



#### Step 3 - Constrained Decoder

##### Step 3.a - Context Building

##### Step 3.b - Generating Logits

##### Step 3.c - Processing Valid IDs

##### Step 3.d - Preparing the Loop and Updating the State

#### Step 4 - Decoding the tokens and Validating the JSON

#### Step 5 - Validating the Function Calls

#### Step 6 - Writing the Output file



### Algorithm Choices and Implementation Strategy



### Design Decisions



## Instructions

### Installation

Due to the fact that this program requires both pydantic, rich and pygame to be
    installed before running, the Makefile create a virtual environment to run
    the program in and install all the dependencies. First run:

    ```
    make install
    ```

Then to enter the virtual environment use the command:

    ```
    source ./matrix/bin/activate
    ```

To leave the virtual environment simply call the command:

    ```
    deactivate
    ```

If you would like to run typing and syntax checks you can run these commands
    (note this is best done before using make install, or after using make
    clean - ensuring you are also outside of the virtual environment):

    ```
    make lint
    make lint-strict
    ```

### Executing the Program

To then execute Fly-in you can call the following terminal commands:

    ```
    make run FILE=<optional_file_path> OUTPUT=<optional_argument>
    ```
In the above template the text in between <> can be replaced with the required
    file directories, and file names. The optional_argument if left blank will
    automatically run the program in an only terminal output mode. If the
    argument 'terminal' is used, the same will happen, while if 'pygame' is
    specified this will run the program to output only using the graphical
    display, and lastly, if 'both' is used, it will then run both the terminal
    output and the graphical display.

### Example Input and Expected Output


## Resources

### Algorithms

- [Pathfinding](https://en.wikipedia.org/wiki/Pathfinding) — general pathfinding info
- [A* algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm) — A* pathfinding info
- [Dijkstra algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) — Dijkstra pathfinding info

### Pygame & Rich

- [Pygame Tutorial](https://www.geeksforgeeks.org/python/pygame-tutorial/) — basic Pygame info
- [Pygame Surface Tutorial](https://www.geeksforgeeks.org/python/pygame-surface/) — info on Surfaces in Pygame
- [Deep Dive Pygame Surface Info](https://www.pygame.org/docs/ref/surface.html#pygame.Surface.copy) - Deeper information about surfaces in pygame
- [Deep Dive Pygame Color Info](https://www.pygame.org/docs/ref/color.html) - Deeper information about colors in pygame
- [Pygame Text Tutorial](https://www.geeksforgeeks.org/python/pygame-working-with-text/) — info on Text in Pygame
- [Pygame Object Tutorial](https://www.geeksforgeeks.org/python/pygame-drawing-objects-and-shapes/) — info on creating Shapes in Pygame
- [Pygame Key Input Tutorial](https://www.geeksforgeeks.org/python/how-to-get-keyboard-input-in-pygame/) — info on using keyboard inputs in Pygame

- [Basics of Rich](https://rich.readthedocs.io/en/latest/introduction.html) — info on using the Rich Python library for colors and terminal output

### AI usage

AI was primarily used to refine some resources as some were leading to dead
    ends or were producing results unrelated to my specific issues. AI was also
    used as a tool to validate some steps such as struggles around looping and
    dead-end errors within the algorithm logic that I had already coded.
