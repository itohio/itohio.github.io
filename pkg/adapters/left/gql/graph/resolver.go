package graph

import (
	"github.com/itohio/itohio.github.io/pkg/config"
	"github.com/itohio/itohio.github.io/pkg/ports"
)

// This file will not be regenerated automatically.
//
// It serves as dependency injection for your app, add any dependencies you require here.

type Resolver struct {
	Cfg *config.Config
	App ports.APIPort
	Db  ports.DbPort
}
