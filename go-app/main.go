package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
)

func main() {
	port := os.Getenv("DATABRICKS_APP_PORT")
	if port == "" {
		port = "8000"
	}

	// Serve static files from the "static" directory
	fs := http.FileServer(http.Dir("./static"))
	http.Handle("/", fs)

	addr := fmt.Sprintf("0.0.0.0:%s", port)
	log.Printf("Go static server listening on %s\n", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
