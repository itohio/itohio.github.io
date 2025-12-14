---
title: "Cobra CLI Framework"
date: 2024-12-12T21:00:00Z
draft: false
description: "Building CLI applications with Cobra in Go"
type: "snippet"
tags: ["go", "golang", "cobra", "cli", "command-line", "go-knowhow"]
category: "go"
---



Cobra is a powerful library for creating modern CLI applications in Go. Used by kubectl, Hugo, GitHub CLI, and many other popular tools. Provides commands, subcommands, flags, and automatic help generation.

## Use Case

Use Cobra when you need to:
- Build complex CLI tools with subcommands
- Provide consistent flag handling
- Generate automatic help and documentation
- Create professional command-line interfaces

## Installation

```bash
go get -u github.com/spf13/cobra@latest
```

## Code

### Basic Structure

```go
package main

import (
    "fmt"
    "os"
    
    "github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
    Use:   "myapp",
    Short: "A brief description of your application",
    Long:  `A longer description that spans multiple lines and likely contains
examples and usage of using your application.`,
}

func main() {
    if err := rootCmd.Execute(); err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(1)
    }
}
```

## Examples

### Example 1: Simple Command with Flags

```go
package main

import (
    "fmt"
    "github.com/spf13/cobra"
)

var (
    verbose bool
    output  string
)

var rootCmd = &cobra.Command{
    Use:   "greet [name]",
    Short: "Greet someone",
    Args:  cobra.ExactArgs(1),
    Run: func(cmd *cobra.Command, args []string) {
        name := args[0]
        greeting := fmt.Sprintf("Hello, %s!", name)
        
        if verbose {
            greeting = fmt.Sprintf("Hello there, %s! Nice to meet you!", name)
        }
        
        if output == "json" {
            fmt.Printf(`{"greeting": "%s"}`, greeting)
        } else {
            fmt.Println(greeting)
        }
    },
}

func init() {
    rootCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Verbose output")
    rootCmd.Flags().StringVarP(&output, "output", "o", "text", "Output format (text|json)")
}

func main() {
    rootCmd.Execute()
}
```

**Usage:**
```bash
$ myapp greet John
Hello, John!

$ myapp greet John --verbose
Hello there, John! Nice to meet you!

$ myapp greet John -o json
{"greeting": "Hello, John!"}
```

### Example 2: Subcommands

```go
package main

import (
    "fmt"
    "github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
    Use:   "app",
    Short: "My application",
}

var initCmd = &cobra.Command{
    Use:   "init",
    Short: "Initialize the application",
    Run: func(cmd *cobra.Command, args []string) {
        fmt.Println("Initializing...")
    },
}

var runCmd = &cobra.Command{
    Use:   "run",
    Short: "Run the application",
    Run: func(cmd *cobra.Command, args []string) {
        fmt.Println("Running...")
    },
}

var configCmd = &cobra.Command{
    Use:   "config",
    Short: "Manage configuration",
}

var configGetCmd = &cobra.Command{
    Use:   "get [key]",
    Short: "Get configuration value",
    Args:  cobra.ExactArgs(1),
    Run: func(cmd *cobra.Command, args []string) {
        fmt.Printf("Getting config: %s\n", args[0])
    },
}

var configSetCmd = &cobra.Command{
    Use:   "set [key] [value]",
    Short: "Set configuration value",
    Args:  cobra.ExactArgs(2),
    Run: func(cmd *cobra.Command, args []string) {
        fmt.Printf("Setting %s = %s\n", args[0], args[1])
    },
}

func init() {
    // Add commands to root
    rootCmd.AddCommand(initCmd)
    rootCmd.AddCommand(runCmd)
    rootCmd.AddCommand(configCmd)
    
    // Add subcommands to config
    configCmd.AddCommand(configGetCmd)
    configCmd.AddCommand(configSetCmd)
}

func main() {
    rootCmd.Execute()
}
```

**Usage:**
```bash
$ app init
Initializing...

$ app config get database.host
Getting config: database.host

$ app config set database.host localhost
Setting database.host = localhost
```

### Example 3: Persistent Flags and PreRun

```go
package main

import (
    "fmt"
    "github.com/spf13/cobra"
)

var (
    cfgFile string
    debug   bool
)

var rootCmd = &cobra.Command{
    Use:   "app",
    Short: "My application",
    PersistentPreRun: func(cmd *cobra.Command, args []string) {
        // Runs before any command
        if debug {
            fmt.Println("Debug mode enabled")
        }
        if cfgFile != "" {
            fmt.Printf("Using config file: %s\n", cfgFile)
        }
    },
}

var serveCmd = &cobra.Command{
    Use:   "serve",
    Short: "Start the server",
    PreRun: func(cmd *cobra.Command, args []string) {
        // Runs before this specific command
        fmt.Println("Preparing to serve...")
    },
    Run: func(cmd *cobra.Command, args []string) {
        fmt.Println("Server started!")
    },
    PostRun: func(cmd *cobra.Command, args []string) {
        // Runs after this specific command
        fmt.Println("Server stopped")
    },
}

func init() {
    // Persistent flags available to all commands
    rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file")
    rootCmd.PersistentFlags().BoolVar(&debug, "debug", false, "debug mode")
    
    rootCmd.AddCommand(serveCmd)
}

func main() {
    rootCmd.Execute()
}
```

### Example 4: Required Flags and Validation

```go
package main

import (
    "fmt"
    "github.com/spf13/cobra"
)

var (
    username string
    password string
    port     int
)

var loginCmd = &cobra.Command{
    Use:   "login",
    Short: "Login to the service",
    PreRunE: func(cmd *cobra.Command, args []string) error {
        // Validation before running
        if port < 1 || port > 65535 {
            return fmt.Errorf("invalid port: %d", port)
        }
        return nil
    },
    RunE: func(cmd *cobra.Command, args []string) error {
        // Use RunE to return errors
        fmt.Printf("Logging in as %s on port %d\n", username, port)
        return nil
    },
}

func init() {
    loginCmd.Flags().StringVarP(&username, "username", "u", "", "Username (required)")
    loginCmd.Flags().StringVarP(&password, "password", "p", "", "Password (required)")
    loginCmd.Flags().IntVar(&port, "port", 8080, "Port number")
    
    // Mark flags as required
    loginCmd.MarkFlagRequired("username")
    loginCmd.MarkFlagRequired("password")
}

func main() {
    rootCmd := &cobra.Command{Use: "app"}
    rootCmd.AddCommand(loginCmd)
    rootCmd.Execute()
}
```

### Example 5: Cobra Generator (Quick Start)

```bash
# Install cobra-cli
go install github.com/spf13/cobra-cli@latest

# Initialize new CLI app
cobra-cli init

# Add commands
cobra-cli add serve
cobra-cli add config
cobra-cli add create

# Project structure created:
# ├── cmd/
# │   ├── root.go
# │   ├── serve.go
# │   ├── config.go
# │   └── create.go
# ├── main.go
# └── go.mod
```

## Common Patterns

### Argument Validation

```go
var cmd = &cobra.Command{
    Use: "example",
    Args: cobra.MinimumNArgs(1),        // At least 1 arg
    // Args: cobra.ExactArgs(2),        // Exactly 2 args
    // Args: cobra.RangeArgs(1, 3),     // Between 1 and 3 args
    // Args: cobra.NoArgs,              // No args allowed
    Run: func(cmd *cobra.Command, args []string) {
        // ...
    },
}
```

### Custom Validation

```go
var cmd = &cobra.Command{
    Use: "example",
    Args: func(cmd *cobra.Command, args []string) error {
        if len(args) < 1 {
            return fmt.Errorf("requires at least 1 arg")
        }
        if args[0] != "valid" {
            return fmt.Errorf("invalid argument: %s", args[0])
        }
        return nil
    },
    Run: func(cmd *cobra.Command, args []string) {
        // ...
    },
}
```

### Version Command

```go
var version = "1.0.0"

var versionCmd = &cobra.Command{
    Use:   "version",
    Short: "Print the version number",
    Run: func(cmd *cobra.Command, args []string) {
        fmt.Printf("Version: %s\n", version)
    },
}

func init() {
    rootCmd.AddCommand(versionCmd)
}
```

## Notes

- Use `PersistentFlags` for flags available to all subcommands
- Use `Flags` for command-specific flags
- `PreRun`, `Run`, `PostRun` provide lifecycle hooks
- Use `RunE` instead of `Run` to return errors
- Cobra automatically generates help and usage

## Gotchas/Warnings

- ⚠️ **Flag parsing**: Flags must come after the command name
- ⚠️ **Required flags**: Use `MarkFlagRequired()` for validation
- ⚠️ **Persistent flags**: Defined on parent, available to children
- ⚠️ **Args validation**: Use built-in validators or custom functions