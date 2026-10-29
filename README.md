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

As mentioned above, both a terminal and graphical approach was implemented in
    this project. The terminal approach uses the base output that is expected
    of which drones move and what hub/connection they moved to during that turn
    with only 1 turn being printed per newline. This was easy to convert into a
    more readable terminal output by using the library Rich, which allowed the
    print outputs to be associated with either the specified or default colour.

Pygame was used as mentioned above for the graphical output. This was made
    pretty simple as pygame is pretty robust with how shapes are created and
    colours can be directly pulled. The simplest way I found to use it was to
    pair its inbuilt sizing and positioning system of drawn objects along with
    the hubs x and y positions. These positions were normalized to start at 0
    where possible and then increase from there to fill the space with all the
    hubs. The connections were then lines that were drawn directly between the
    hub x and y positions. The image of hubs and connections were then copied
    so that they could be blit'ed onto the screen to remove drones that had
    moved and then redrawing the drone locations through their current location
    using that x and y value, matching to the center of the hub rectangles.

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

#### Pygame

While running pygame, you can press 'esc' key or the close terminal UI button
    and the pygame will close. The program will continue running and produce
    terminal output instead. You can force close the terminal output/program
    early like normal using 'Ctrl+c' on the terminal.

Once the program has finished moving all drones, Pygame will remain open. You
    can close the program using the same commands as above or you can press
    'space' to close the program.

### Example Input and Expected Output

For the files, I would suggest using the ones provided or creating a txt file
    with inputs like this:

    ```
    nb_drones: 2

    start_hub: start -1 0 [color=green]
    hub: waypoint1 1 1 [zone=restricted color=blue]
    hub: waypoint2 2 1 [zone=priority color=blue]
    hub: waypoint3 3 1 [zone=restricted color=blue]
    hub: waypointA 1 2 [zone=priority color=blue]
    hub: waypointB 2 2 [zone=priority color=blue]
    hub: waypointC 3 2 [color=blue]
    end_hub: goal 3 3 [color=red]

    connection: start-waypoint1
    connection: start-waypoint2
    connection: start-waypoint3
    connection: waypoint1-waypointA
    connection: waypoint1-waypointB
    connection: waypoint2-waypointA
    connection: waypoint2-waypointB
    connection: waypoint2-waypointC
    connection: waypoint3-waypointB
    connection: waypoint3-waypointC
    connection: waypointA-waypointB
    connection: waypointB-waypointC
    connection: waypointB-goal
    connection: waypointC-goal
    ```

Lastly, you can then run the below command if all files are in place:

    ```
    make run FILE=./maps/easy/test.txt OUTPUT=pygame
    ```

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
