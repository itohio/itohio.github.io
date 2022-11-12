package gql

//go:generate gqlgen

import (
	"net"
	"net/http"
	"time"

	"github.com/itohio/itohio.github.io/pkg/adapters/left/auth"
	"github.com/itohio/itohio.github.io/pkg/adapters/left/gql/graph"
	"github.com/itohio/itohio.github.io/pkg/adapters/left/gql/graph/generated"
	"github.com/itohio/itohio.github.io/pkg/config"
	"github.com/itohio/itohio.github.io/pkg/ports"

	"github.com/99designs/gqlgen/graphql/handler"
	"github.com/99designs/gqlgen/graphql/playground"
	"github.com/go-chi/chi/middleware"
	"github.com/go-chi/chi/v5"
	"github.com/rs/cors"
)

type Adapter struct {
	cfg    *config.Config
	app    ports.APIPort
	listen net.Listener
	router http.Handler
	routes map[string]http.Handler
}

func New(cfg *config.Config, app ports.APIPort, db ports.DbPort, routes map[string]http.Handler) (*Adapter, error) {
	ret := &Adapter{
		cfg:    cfg,
		app:    app,
		routes: routes,
	}

	ln, err := net.Listen("tcp", cfg.Server.Address)
	if err != nil {
		return nil, err
	}
	ret.listen = ln

	authMiddleware := auth.NewMiddleware(cfg.Auth.Domain, cfg.Auth.Audience, cfg.Debug)
	router := chi.NewRouter()
	_ = authMiddleware
	srv := handler.NewDefaultServer(
		generated.NewExecutableSchema(
			generated.Config{
				Resolvers: &graph.Resolver{
					Cfg: cfg,
					App: app,
					Db:  db,
				},
				Directives: generated.DirectiveRoot{
					IsAuthenticated: IsAuthenticated,
					HasScope:        HasScope,
				},
			},
		),
	)

	router.Handle("/", authMiddleware.Handler(srv))
	router.HandleFunc("/play", playground.Handler("GraphQL playground", "/api"))

	ret.router = router

	return ret, nil
}

func (a *Adapter) Run() error {
	router := chi.NewRouter()
	router.Use(middleware.Logger)
	router.Use(middleware.Recoverer)

	cors := cors.New(cors.Options{
		AllowedOrigins:   a.cfg.Server.Origins,
		AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-CSRF-Token"},
		ExposedHeaders:   []string{"Link", "Authorization"},
		AllowCredentials: true,
		MaxAge:           300, // Maximum value not ignored by any of major browsers
	})
	router.Use(cors.Handler)

	router.Mount("/api", a.router)
	for k, v := range a.routes {
		router.Mount(k, v)
	}

	srv := &http.Server{
		Handler: router,
		Addr:    a.cfg.Server.Address,
		// Good practice: enforce timeouts for servers you create!
		WriteTimeout: 15 * time.Second,
		ReadTimeout:  15 * time.Second,
	}

	if err := srv.Serve(a.listen); err != nil {
		return err
	}

	return nil
}
