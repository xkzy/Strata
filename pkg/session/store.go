// pkg/session/store.go - Disk-backed thread-safe store for chat sessions.
package session

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Store manages disk-persisted chat sessions in a target directory.
type Store struct {
	mu  sync.RWMutex
	dir string
}

// NewStore initializes a session store at the given directory.
func NewStore(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create session dir %q: %w", dir, err)
	}
	return &Store{dir: dir}, nil
}

// Dir returns the underlying session storage directory path.
func (s *Store) Dir() string {
	return s.dir
}

func (s *Store) filePath(id string) (string, error) {
	clean, err := SanitizeID(id)
	if err != nil {
		return "", err
	}
	return filepath.Join(s.dir, clean+".json"), nil
}

// List returns summaries of all saved sessions, ordered newest to oldest by updated_at.
func (s *Store) List() ([]SessionSummary, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entries, err := os.ReadDir(s.dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []SessionSummary{}, nil
		}
		return nil, fmt.Errorf("read session dir: %w", err)
	}

	summaries := make([]SessionSummary, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") || strings.Contains(entry.Name(), ".tmp.") {
			continue
		}
		id := strings.TrimSuffix(entry.Name(), ".json")
		sess, err := s.readSessionFile(filepath.Join(s.dir, entry.Name()))
		if err != nil {
			continue // skip corrupted or unreadable files gracefully
		}
		if sess.ID == "" {
			sess.ID = id
		}
		summaries = append(summaries, sess.Summary())
	}

	// Sort descending by UpdatedAt (fallback to CreatedAt)
	sort.Slice(summaries, func(i, j int) bool {
		tI := summaries[i].UpdatedAt
		if tI == 0 {
			tI = summaries[i].CreatedAt
		}
		tJ := summaries[j].UpdatedAt
		if tJ == 0 {
			tJ = summaries[j].CreatedAt
		}
		return tI > tJ
	})

	return summaries, nil
}

// Get loads a specific session by ID. Returns os.ErrNotExist if not found.
func (s *Store) Get(id string) (*Session, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	path, err := s.filePath(id)
	if err != nil {
		return nil, err
	}

	sess, err := s.readSessionFile(path)
	if err != nil {
		return nil, err
	}
	if sess.ID == "" {
		clean, _ := SanitizeID(id)
		sess.ID = clean
	}
	return sess, nil
}

func (s *Store) readSessionFile(path string) (*Session, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var sess Session
	if err := json.Unmarshal(data, &sess); err != nil {
		return nil, fmt.Errorf("unmarshal session file %s: %w", filepath.Base(path), err)
	}
	return &sess, nil
}

// Save atomically writes a session to disk.
func (s *Store) Save(sess *Session) error {
	if sess == nil {
		return errors.New("cannot save nil session")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if sess.ID == "" {
		sess.ID = GenerateID()
	}
	cleanID, err := SanitizeID(sess.ID)
	if err != nil {
		sess.ID = GenerateID()
		cleanID, _ = SanitizeID(sess.ID)
	}
	sess.ID = cleanID

	now := time.Now().UnixMilli()
	if sess.CreatedAt == 0 {
		sess.CreatedAt = now
	}
	sess.UpdatedAt = now

	if sess.Title == "" {
		sess.Title = "New Chat"
	}
	if sess.Messages == nil {
		sess.Messages = []Message{}
	}

	data, err := json.MarshalIndent(sess, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal session %s: %w", sess.ID, err)
	}

	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return fmt.Errorf("ensure session dir: %w", err)
	}

	// Atomic write via temporary file
	var randBytes [6]byte
	_, _ = rand.Read(randBytes[:])
	tmpName := fmt.Sprintf("%s.tmp.%s", sess.ID, hex.EncodeToString(randBytes[:]))
	tmpPath := filepath.Join(s.dir, tmpName)
	targetPath := filepath.Join(s.dir, sess.ID+".json")

	if err := os.WriteFile(tmpPath, data, 0o644); err != nil {
		return fmt.Errorf("write temp session file %s: %w", tmpPath, err)
	}

	if err := os.Rename(tmpPath, targetPath); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("atomic rename session file: %w", err)
	}

	return nil
}

// Delete removes a session file by ID. Returns true if the file was deleted.
func (s *Store) Delete(id string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	path, err := s.filePath(id)
	if err != nil {
		return false, err
	}

	if err := os.Remove(path); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return false, fmt.Errorf("remove session %s: %w", id, err)
	}
	return true, nil
}

// Clear removes all session files in the storage directory.
func (s *Store) Clear() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	entries, err := os.ReadDir(s.dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("read session dir for clear: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if strings.HasSuffix(entry.Name(), ".json") || strings.Contains(entry.Name(), ".tmp.") {
			_ = os.Remove(filepath.Join(s.dir, entry.Name()))
		}
	}
	return nil
}

// Fork creates a new session branch from an existing session, copying messages up to messageIndex.
// If messageIndex <= 0 or > len(messages), all messages are copied.
func (s *Store) Fork(parentID string, newID string, messageIndex int, newTitle string) (*Session, error) {
	parent, err := s.Get(parentID)
	if err != nil {
		return nil, fmt.Errorf("parent session %q: %w", parentID, err)
	}

	if newID == "" {
		newID = GenerateID()
	}
	cleanID, err := SanitizeID(newID)
	if err != nil {
		cleanID = GenerateID()
	}

	if newTitle == "" {
		newTitle = fmt.Sprintf("%s (Fork)", parent.Title)
	}

	var forkMessages []Message
	if messageIndex > 0 && messageIndex < len(parent.Messages) {
		forkMessages = make([]Message, messageIndex)
		copy(forkMessages, parent.Messages[:messageIndex])
	} else {
		forkMessages = make([]Message, len(parent.Messages))
		copy(forkMessages, parent.Messages)
	}

	forked := &Session{
		ID:        cleanID,
		Title:     newTitle,
		CreatedAt: time.Now().UnixMilli(),
		UpdatedAt: time.Now().UnixMilli(),
		Model:     parent.Model,
		Messages:  forkMessages,
		Pinned:    false,
		Tags:      append([]string{}, parent.Tags...),
	}

	if err := s.Save(forked); err != nil {
		return nil, fmt.Errorf("save forked session: %w", err)
	}

	return forked, nil
}

// Search queries sessions by title, tags, and message text matching the query string.
func (s *Store) Search(query string) ([]SessionSummary, error) {
	query = strings.ToLower(strings.TrimSpace(query))
	if query == "" {
		return s.List()
	}

	all, err := s.List()
	if err != nil {
		return nil, err
	}

	var matched []SessionSummary
	for _, summary := range all {
		if strings.Contains(strings.ToLower(summary.Title), query) {
			matched = append(matched, summary)
			continue
		}
		// Check tags
		tagMatch := false
		for _, tag := range summary.Tags {
			if strings.Contains(strings.ToLower(tag), query) {
				matched = append(matched, summary)
				tagMatch = true
				break
			}
		}
		if tagMatch {
			continue
		}
		// Check messages
		sess, err := s.Get(summary.ID)
		if err == nil {
			for _, m := range sess.Messages {
				if strings.Contains(strings.ToLower(m.Text), query) {
					matched = append(matched, summary)
					break
				}
			}
		}
	}

	return matched, nil
}
