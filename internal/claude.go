package internal

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/google/uuid"
)

// Claude API 格式
type ClaudeContent struct {
	Type   string                 `json:"type"`
	Text   string                 `json:"text,omitempty"`
	Source map[string]interface{} `json:"source,omitempty"`
}

type ClaudeMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type ClaudeRequest struct {
	Model     string          `json:"model"`
	Messages  []ClaudeMessage `json:"messages"`
	MaxTokens int             `json:"max_tokens"`
	Stream    bool            `json:"stream,omitempty"`
	System    json.RawMessage `json:"system,omitempty"`
}

type ClaudeStreamResponse struct {
	Type  string                 `json:"type"`
	Index int                    `json:"index,omitempty"`
	Delta map[string]interface{} `json:"delta,omitempty"`
}

type ClaudeResponse struct {
	ID      string          `json:"id"`
	Type    string          `json:"type"`
	Role    string          `json:"role"`
	Content []ClaudeContent `json:"content"`
	Model   string          `json:"model"`
	Usage   map[string]int  `json:"usage"`
}

// 转换 Claude 消息为内部格式
func convertClaudeMessages(claudeMessages []ClaudeMessage) []Message {
	var messages []Message
	for _, cm := range claudeMessages {
		var content interface{}
		
		// 尝试解析为字符串
		var textContent string
		if err := json.Unmarshal(cm.Content, &textContent); err == nil {
			content = textContent
		} else {
			// 尝试解析为数组
			var arrayContent []map[string]interface{}
			if err := json.Unmarshal(cm.Content, &arrayContent); err == nil {
				var parts []interface{}
				for _, item := range arrayContent {
					itemType, _ := item["type"].(string)
					if itemType == "text" {
						parts = append(parts, map[string]interface{}{
							"type": "text",
							"text": item["text"],
						})
					} else if itemType == "image" {
						if source, ok := item["source"].(map[string]interface{}); ok {
							if sourceType, _ := source["type"].(string); sourceType == "base64" {
								mediaType, _ := source["media_type"].(string)
								data, _ := source["data"].(string)
								url := fmt.Sprintf("data:%s;base64,%s", mediaType, data)
								parts = append(parts, map[string]interface{}{
									"type": "image_url",
									"image_url": map[string]interface{}{
										"url": url,
									},
								})
							} else if sourceType == "url" {
								url, _ := source["url"].(string)
								parts = append(parts, map[string]interface{}{
									"type": "image_url",
									"image_url": map[string]interface{}{
										"url": url,
									},
								})
							}
						}
					}
				}
				content = parts
			}
		}
		
		messages = append(messages, Message{
			Role:    cm.Role,
			Content: content,
		})
	}
	return messages
}

// 映射 Claude 模型到内部模型
func mapClaudeModel(claudeModel string) string {
	// Claude 模型映射到 GLM 模型
	if strings.Contains(claudeModel, "opus") || strings.Contains(claudeModel, "sonnet") {
		return "GLM-4.6"
	}
	return "GLM-4.5"
}

// 解析 system 字段，支持字符串或数组格式
func parseSystemMessage(systemRaw json.RawMessage) string {
	if len(systemRaw) == 0 {
		return ""
	}

	// 尝试解析为字符串
	var systemStr string
	if err := json.Unmarshal(systemRaw, &systemStr); err == nil {
		return systemStr
	}

	// 尝试解析为数组
	var systemArray []map[string]interface{}
	if err := json.Unmarshal(systemRaw, &systemArray); err == nil {
		var parts []string
		for _, item := range systemArray {
			if itemType, ok := item["type"].(string); ok && itemType == "text" {
				if text, ok := item["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
		return strings.Join(parts, "\n")
	}

	return ""
}

func HandleClaudeChatCompletions(w http.ResponseWriter, r *http.Request) {
	// 处理 CORS 预检请求
	if r.Method == "OPTIONS" {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, x-api-key, Authorization, anthropic-version")
		w.WriteHeader(http.StatusOK)
		return
	}

	// 设置 CORS 头
	w.Header().Set("Access-Control-Allow-Origin", "*")
	
	LogDebug("[Claude] Method: %s, Headers: x-api-key=%s, Authorization=%s", r.Method, r.Header.Get("x-api-key"), r.Header.Get("Authorization"))
	
	apiKey := r.Header.Get("x-api-key")
	if apiKey == "" {
		apiKey = strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	}
	if apiKey == "" {
		LogError("[Claude] Missing API key")
		w.Header().Set("Content-Type", "application/json")
		http.Error(w, `{"error":{"type":"authentication_error","message":"Missing API key"}}`, http.StatusUnauthorized)
		return
	}
	
	LogDebug("[Claude] Using API key: %s...", apiKey[:min(10, len(apiKey))])

	if apiKey == "free" {
		anonymousToken, err := GetAnonymousToken()
		if err != nil {
			LogError("Failed to get anonymous token: %v", err)
			http.Error(w, `{"error":{"type":"api_error","message":"Failed to get token"}}`, http.StatusInternalServerError)
			return
		}
		apiKey = anonymousToken
	}

	var req ClaudeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		LogError("[Claude] Invalid JSON: %v", err)
		w.Header().Set("Content-Type", "application/json")
		http.Error(w, `{"error":{"type":"invalid_request_error","message":"Invalid JSON"}}`, http.StatusBadRequest)
		return
	}

	LogDebug("[Claude] Request: model=%s, messages=%d, stream=%v", req.Model, len(req.Messages), req.Stream)

	internalModel := mapClaudeModel(req.Model)
	messages := convertClaudeMessages(req.Messages)

	// 处理 system 消息
	systemMsg := parseSystemMessage(req.System)
	if systemMsg != "" {
		messages = append([]Message{{Role: "system", Content: systemMsg}}, messages...)
	}

	resp, _, err := makeUpstreamRequest(apiKey, messages, internalModel)
	if err != nil {
		LogError("Upstream request failed: %v", err)
		http.Error(w, `{"error":{"type":"api_error","message":"Upstream error"}}`, http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		http.Error(w, `{"error":{"type":"api_error","message":"Upstream error"}}`, resp.StatusCode)
		return
	}

	completionID := fmt.Sprintf("msg_%s", uuid.New().String()[:24])

	if req.Stream {
		handleClaudeStreamResponse(w, resp.Body, completionID, req.Model)
	} else {
		handleClaudeNonStreamResponse(w, resp.Body, completionID, req.Model)
	}
}

func handleClaudeStreamResponse(w http.ResponseWriter, body io.ReadCloser, completionID, model string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// 发送 message_start
	startEvent := map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id":    completionID,
			"type":  "message",
			"role":  "assistant",
			"model": model,
			"usage": map[string]interface{}{
				"input_tokens":  0,
				"output_tokens": 0,
			},
		},
	}
	data, _ := json.Marshal(startEvent)
	fmt.Fprintf(w, "event: message_start\ndata: %s\n\n", data)
	flusher.Flush()

	// 发送 content_block_start
	blockStart := map[string]interface{}{
		"type":         "content_block_start",
		"index":        0,
		"content_block": map[string]interface{}{"type": "text", "text": ""},
	}
	data, _ = json.Marshal(blockStart)
	fmt.Fprintf(w, "event: content_block_start\ndata: %s\n\n", data)
	flusher.Flush()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	searchRefFilter := NewSearchRefFilter()
	thinkingFilter := &ThinkingFilter{}
	pendingSourcesMarkdown := ""

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}

		var upstream UpstreamData
		if err := json.Unmarshal([]byte(payload), &upstream); err != nil {
			continue
		}

		if upstream.Data.Phase == "done" {
			break
		}

		if upstream.Data.Phase == "thinking" && upstream.Data.DeltaContent != "" {
			if pendingSourcesMarkdown != "" {
				pendingSourcesMarkdown = ""
			}
			thinkingFilter.ProcessThinking(upstream.Data.DeltaContent)
			continue
		}

		if upstream.Data.EditContent != "" && IsSearchResultContent(upstream.Data.EditContent) {
			if results := ParseSearchResults(upstream.Data.EditContent); len(results) > 0 {
				searchRefFilter.AddSearchResults(results)
				pendingSourcesMarkdown = searchRefFilter.GetSearchResultsMarkdown()
			}
			continue
		}

		if upstream.Data.EditContent != "" && IsSearchToolCall(upstream.Data.EditContent, upstream.Data.Phase) {
			continue
		}

		if pendingSourcesMarkdown != "" {
			delta := map[string]interface{}{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]interface{}{"type": "text_delta", "text": pendingSourcesMarkdown},
			}
			data, _ := json.Marshal(delta)
			fmt.Fprintf(w, "event: content_block_delta\ndata: %s\n\n", data)
			flusher.Flush()
			pendingSourcesMarkdown = ""
		}

		content := ""
		if upstream.Data.Phase == "answer" && upstream.Data.DeltaContent != "" {
			content = upstream.Data.DeltaContent
		} else if upstream.Data.Phase == "answer" && upstream.Data.EditContent != "" {
			if strings.Contains(upstream.Data.EditContent, "</details>") {
				thinkingFilter.ExtractCompleteThinking(upstream.Data.EditContent)
				if idx := strings.Index(upstream.Data.EditContent, "</details>\n"); idx != -1 {
					content = upstream.Data.EditContent[idx+len("</details>\n"):]
				}
			}
		} else if (upstream.Data.Phase == "other" || upstream.Data.Phase == "tool_call") && upstream.Data.EditContent != "" {
			content = upstream.Data.EditContent
		}

		if content == "" {
			continue
		}

		content = searchRefFilter.Process(content)
		if content == "" {
			continue
		}

		delta := map[string]interface{}{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]interface{}{"type": "text_delta", "text": content},
		}
		data, _ = json.Marshal(delta)
		fmt.Fprintf(w, "event: content_block_delta\ndata: %s\n\n", data)
		flusher.Flush()
	}

	if remaining := searchRefFilter.Flush(); remaining != "" {
		delta := map[string]interface{}{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]interface{}{"type": "text_delta", "text": remaining},
		}
		data, _ := json.Marshal(delta)
		fmt.Fprintf(w, "event: content_block_delta\ndata: %s\n\n", data)
		flusher.Flush()
	}

	// 发送 content_block_stop
	blockStop := map[string]interface{}{"type": "content_block_stop", "index": 0}
	data, _ = json.Marshal(blockStop)
	fmt.Fprintf(w, "event: content_block_stop\ndata: %s\n\n", data)
	flusher.Flush()

	// 发送 message_delta
	messageDelta := map[string]interface{}{
		"type": "message_delta",
		"delta": map[string]interface{}{
			"stop_reason": "end_turn",
		},
		"usage": map[string]interface{}{
			"output_tokens": 0,
		},
	}
	data, _ = json.Marshal(messageDelta)
	fmt.Fprintf(w, "event: message_delta\ndata: %s\n\n", data)
	flusher.Flush()

	// 发送 message_stop
	messageStop := map[string]interface{}{"type": "message_stop"}
	data, _ = json.Marshal(messageStop)
	fmt.Fprintf(w, "event: message_stop\ndata: %s\n\n", data)
	flusher.Flush()
}

func handleClaudeNonStreamResponse(w http.ResponseWriter, body io.ReadCloser, completionID, model string) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	var chunks []string
	thinkingFilter := &ThinkingFilter{}
	searchRefFilter := NewSearchRefFilter()
	pendingSourcesMarkdown := ""

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}

		var upstream UpstreamData
		if err := json.Unmarshal([]byte(payload), &upstream); err != nil {
			continue
		}

		if upstream.Data.Phase == "done" {
			break
		}

		if upstream.Data.Phase == "thinking" && upstream.Data.DeltaContent != "" {
			thinkingFilter.ProcessThinking(upstream.Data.DeltaContent)
			continue
		}

		if upstream.Data.EditContent != "" && IsSearchResultContent(upstream.Data.EditContent) {
			if results := ParseSearchResults(upstream.Data.EditContent); len(results) > 0 {
				searchRefFilter.AddSearchResults(results)
				pendingSourcesMarkdown = searchRefFilter.GetSearchResultsMarkdown()
			}
			continue
		}

		if upstream.Data.EditContent != "" && IsSearchToolCall(upstream.Data.EditContent, upstream.Data.Phase) {
			continue
		}

		if pendingSourcesMarkdown != "" {
			chunks = append(chunks, pendingSourcesMarkdown)
			pendingSourcesMarkdown = ""
		}

		content := ""
		if upstream.Data.Phase == "answer" && upstream.Data.DeltaContent != "" {
			content = upstream.Data.DeltaContent
		} else if upstream.Data.Phase == "answer" && upstream.Data.EditContent != "" {
			if strings.Contains(upstream.Data.EditContent, "</details>") {
				thinkingFilter.ExtractCompleteThinking(upstream.Data.EditContent)
				if idx := strings.Index(upstream.Data.EditContent, "</details>\n"); idx != -1 {
					content = upstream.Data.EditContent[idx+len("</details>\n"):]
				}
			}
		} else if (upstream.Data.Phase == "other" || upstream.Data.Phase == "tool_call") && upstream.Data.EditContent != "" {
			content = upstream.Data.EditContent
		}

		if content != "" {
			chunks = append(chunks, content)
		}
	}

	fullContent := strings.Join(chunks, "")
	fullContent = searchRefFilter.Process(fullContent) + searchRefFilter.Flush()

	response := ClaudeResponse{
		ID:    completionID,
		Type:  "message",
		Role:  "assistant",
		Model: model,
		Content: []ClaudeContent{{
			Type: "text",
			Text: fullContent,
		}},
		Usage: map[string]int{
			"input_tokens":  0,
			"output_tokens": 0,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
