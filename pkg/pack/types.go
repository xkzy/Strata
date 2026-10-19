package pack

// PackOptions configures model packaging
type PackOptions struct {
	GGUFPath   string
	OutDir     string
	BasePack   string
	ExpertsBin bool
	CompatBF16 bool
}

// LayerExperts tracks expert tensor locations and blob dimensions for a layer
type LayerExperts struct {
	Layer      int
	GUType     uint32
	DType      uint32
	Offset     uint64
	BlobBytes  uint64
	GateOffset uint64
	UpOffset   uint64
	DownOffset uint64
	ShardName  string
}

// ConversionRecord documents any tensor format narrowing/widening
type ConversionRecord struct {
	Tensor string `json:"tensor"`
	Src    string `json:"src"`
	Dst    string `json:"dst"`
	Method string `json:"method"`
}
